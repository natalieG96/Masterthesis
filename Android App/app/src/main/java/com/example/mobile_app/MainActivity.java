package com.example.mobile_app;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.view.View;

import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

import static java.lang.Math.min;


/*
Wird die App auf ein Smartphone geladen bzw. mit dem Simulator gestartet, ist nur ein ein Button rechts unten zu sehen. Auf diesen werden alle Aktionen gestartet.
Das heißt, der TF Interpreter wird geladen und somit das Modell mit den Gewichtungen in den Speicher geholt. Damit können nun Vorhersagen getroffen werden.
Die Bilder zum Vorhersagen sind momentan nocht fest in den Assets festgelegt. Der Text auf der Oberfläche der App zeigt die zeitlichen Aspekte der Inferenz bzw. der Metriken, je nach dem was ausgegeben werden soll.
 Die Metriken wurden für den mobilen Ansatz eigens entwickelt, sowie Hilfsfunktionen dafür.
 */
public class MainActivity extends AppCompatActivity {

    Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = findViewById(R.id.fab);
        TextView timetext = findViewById(R.id.timetext);

        try{
            tflite = new Interpreter(loadModelFile());
        }catch (Exception e){
            e.printStackTrace();
        }


        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                float[][][][] prediction = inference(timetext);

                long startTime = System.currentTimeMillis();

                InputStream stream = getFilePaths();
                float[] npyData = null;
                try {
                    Npy test = new Npy(stream);
                    npyData = test.floatElements();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                float[][][] arr =  flattenArrayTo3D(npyData);

                //double iou = intersectionOverUnion(prediction[0], arr);
                double hausdorffer = hausdorff_distance(prediction[0], arr);
                long endTime = System.currentTimeMillis();
                double seconds = ((double)endTime - startTime) / 1000;
                timetext.setText(String.valueOf(seconds) +"s");
            }

        });
    }



    private InputStream getFilePaths(){
        InputStream stream = null;
        try {
            stream = this.getAssets().open("flattenairplane_3.npy");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stream;
    }

    public float[][][][] inference(TextView timetext){
        //long startTime = System.currentTimeMillis();
        InputStream image_stream = null;
        try {
            image_stream = this.getAssets().open("12.png");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(image_stream);

        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        int imageSizeY = imageShape[1];
        int imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        TensorImage tensorImage = new TensorImage(imageDataType);
        tensorImage.load(bitmap);

        int cropSize = min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .build();
        tensorImage = imageProcessor.process(tensorImage);


        float[][][][] outputValue = new float[1][32][32][32];
        ByteBuffer bla = tensorImage.getBuffer();
        tflite.run(tensorImage.getBuffer(), outputValue);

        //long endTime = System.currentTimeMillis();
        //double seconds = ((double)endTime - startTime) / 1000;
        //timetext.setText(String.valueOf(seconds) +"s");

        return outputValue;
    }

    private MappedByteBuffer loadModelFile() throws IOException{
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model_2.tflite");
        FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffSets = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffSets, declaredLength);
    }

    private double hausdorff_distance(final float[][][] aa, final float[][][] bb){

        float[][] a_new = convertArray3DIn2D(aa);
        float[][] b_new = convertArray3DIn2D(bb);

        double maxDistAB = 0;

        for (int a = 0; a < a_new.length; a++){
            int minB = 1000000;
            for (int b = 0; b < b_new.length; b++) {

                int dx = (Math.round(a_new[a][0]) - Math.round(b_new[b][0]));
                int dy = (Math.round(b_new[b][0]) - Math.round(a_new[a][0]));
                int tmpDist = dx*dx + dy*dy;

                if (tmpDist < minB){
                    minB = tmpDist;
                }

                if ( tmpDist == 0 ){
                    break; // can't be better than 0
                }
            }
            maxDistAB += minB;
        }
        return maxDistAB;
    }

    private double intersectionOverUnion(final float[][][] aa, final float[][][] bb){
        double intersection_value = 0;
        int intersected = 0;
        int union = 0;
        for (int i = 0; i < aa.length; i++)
        {
            for (int a = 0; a < aa[i].length; a++)
            {
                for (int b = 0; b < aa[i][a].length; b++)
                {
                    if(aa[i][a][b] >= 0.99f && bb[i][a][b] >= 0.99f){
                        intersected++;
                        union++;
                    }
                    if(aa[i][a][b] >= 0.99f || bb[i][a][b] >= 0.99f){
                        union++;
                    }
                }
            }
        }
        intersection_value = intersected / union;

        return intersection_value;
    }

    private float[][] convertArray3DIn2D(final float[][][] arr){
        float[][] new_arr = new float[258][arr.length * arr.length];
        int counter = 0;
        for (int i = 0; i < arr.length; i++)
        {
            for (int a = 0; a < arr[i].length; a++)
            {
                for (int b = 0; b < arr[i][a].length; b++)
                {
                    if (arr[i][a][b] >= 0.99){
                        new_arr[i][counter] = 1;
                    }
                    else{
                        new_arr[i][counter] = 0;
                    }
                    counter ++;
                }
            }
            counter=0;
        }
        return new_arr;
    }

    private float[][][] flattenArrayTo3D(float[] arr){
        int shape = 258;
        float[][][] newarr = new float[shape][shape][shape];
        int counter = 0;
        for (int i = 0; i < shape; i++)
        {
            for (int a = 0; a < shape; a++)
            {
                for (int b = 0; b <shape; b++)
                {
                    newarr[i][a][b] = arr[counter];
                    counter++;
                }
            }
        }
        return newarr;
    }




    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}