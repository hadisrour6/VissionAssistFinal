package com.example.vissionassistfinal;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class HowToUseActivity extends AppCompatActivity {
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

        // Back Button functionality
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish(); // This closes the current activity and returns to the previous screen
            }
        });
    }
}
