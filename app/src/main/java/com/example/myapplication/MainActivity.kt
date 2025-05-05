package com.example.myapplication

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import org.pytorch.Module

class MainActivity : AppCompatActivity() {

    private val REQUEST_IMAGE_CAPTURE = 1
    private lateinit var imageView: ImageView
    private lateinit var module: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), 100)
        }

        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), 101)
        }

        module = Module.load(assetFilePath(this, "best.torchscript"))

        val captureButton: Button = findViewById(R.id.capture_button)
        imageView = findViewById(R.id.image_view)

        captureButton.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as? Bitmap ?: return
            saveImageToMediaStore(imageBitmap)

            val resizedBitmap = Bitmap.createScaledBitmap(imageBitmap, 640, 640, true)

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val shape = outputTensor.shape()  // Should be [N, 6]
            val outputs = outputTensor.dataAsFloatArray

            val mutableBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)
            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 3f
            }

            val numDetections = shape[0].toInt()
            val maxBoxes = 5
            var boxCount = 0

            for (i in 0 until numDetections) {
                val offset = i * 6
                val x1 = outputs[offset]
                val y1 = outputs[offset + 1]
                val x2 = outputs[offset + 2]
                val y2 = outputs[offset + 3]
                val conf = outputs[offset + 4]
                val cls = outputs[offset + 5]

                if (conf > 0.5f) {
                    // Optionally scale to original image size
                    // val scaleX = imageBitmap.width / 640f
                    // val scaleY = imageBitmap.height / 640f
                    // canvas.drawRect(x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY, paint)

                    canvas.drawRect(x1, y1, x2, y2, paint)
                    boxCount++
                }

                if (boxCount >= maxBoxes) break
            }

            imageView.setImageBitmap(mutableBitmap)
        }
    }

    private fun saveImageToMediaStore(bitmap: Bitmap) {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "Captured Image")
            put(MediaStore.Images.Media.DESCRIPTION, "Image captured from camera")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        }

        val imageUri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        imageUri?.let { uri ->
            contentResolver.openOutputStream(uri)?.let { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                outputStream.close()
            }
        }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }
}
