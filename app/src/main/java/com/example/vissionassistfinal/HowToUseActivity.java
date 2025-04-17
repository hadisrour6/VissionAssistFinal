package com.example.vissionassistfinal;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import java.util.Locale;

public class HowToUseActivity extends AppCompatActivity implements TextToSpeech.OnInitListener {
    private TextToSpeech tts;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_how_to_use);

        TextView howToUseText = findViewById(R.id.howToUseText);
        Button backButton = findViewById(R.id.backButton);

        howToUseText.setText("1. Press 'Start' to open the camera.\n\n" +
                "2. The app detects objects and gives audio feedback.\n\n" +
                "3. Move the camera to scan surroundings.\n\n" +
                "4. Use headphones for better guidance.");

        tts = new TextToSpeech(this, this);

        // Back Button functionality
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish(); // This closes the current activity and returns to the previous screen
            }
        });
    }
    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS) {
            // Set language, etc.
            tts.setLanguage(Locale.US);
            speakInstructions();
        }
    }
    private void speakInstructions() {
        // Replace with your actual instructions
        String instructions = "Here is how to use the app..." +
                " first press 'Start' to open the camera, " +
                " the app will start detecting objects and gives audio feedback. " +
                " Use headphones for better guidance";
        tts.speak(instructions, TextToSpeech.QUEUE_FLUSH, null, "HOW_TO_USE_TTS");
    }
    @Override
    protected void onDestroy() {
        // Clean up TTS
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        super.onDestroy();
    }
}
