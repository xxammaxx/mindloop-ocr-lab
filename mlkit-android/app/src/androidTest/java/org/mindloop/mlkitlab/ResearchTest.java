package org.mindloop.mlkitlab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ResearchTest {
    static int clamp(int value) { return Math.max(0, Math.min(255, value)); }

    // Same BT.601 Y/V/U arithmetic and replicated even padding as Umliva's adapter.
    static byte[] nv21(int[] rgb, int w, int h) {
        int ew = (w + 1) & ~1, eh = (h + 1) & ~1;
        byte[] data = new byte[ew * eh * 3 / 2];
        int uv = ew * eh;
        for (int y = 0; y < eh; y += 2) {
            for (int x = 0; x < ew; x += 2) {
                int sr = 0, sg = 0, sb = 0;
                for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                    int c = rgb[Math.min(y + dy, h - 1) * w + Math.min(x + dx, w - 1)];
                    int r = (c >> 16) & 255, g = (c >> 8) & 255, b = c & 255;
                    sr += r; sg += g; sb += b;
                    data[(y + dy) * ew + x + dx] = (byte)clamp(((66*r + 129*g + 25*b + 128) >> 8) + 16);
                }
                int r = sr / 4, g = sg / 4, b = sb / 4;
                data[uv++] = (byte)clamp(((112*r - 94*g - 18*b + 128) >> 8) + 128);
                data[uv++] = (byte)clamp(((-38*r - 74*g + 112*b + 128) >> 8) + 128);
            }
        }
        return data;
    }

    @Test public void oddRasterPaddingRetainsPixelsAndVUOrder() {
        byte[] frame = nv21(new int[]{0xffff0000}, 1, 1);
        assertArrayEquals(new byte[]{82,82,82,82,(byte)240,90}, frame);
    }

    @Test public void runPairedResearch() throws Exception {
        Context assets = InstrumentationRegistry.getInstrumentation().getContext();
        Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
        JSONArray index;
        try (InputStream input = assets.getAssets().open("index.json")) {
            index = new JSONArray(new String(input.readAllBytes(), StandardCharsets.UTF_8));
        }
        assertTrue("empty experiment", index.length() > 0);
        JSONArray rows = new JSONArray();
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        try {
            for (int i = 0; i < index.length(); i++) {
                JSONObject item = index.getJSONObject(i);
                Bitmap image;
                try (InputStream input = assets.getAssets().open(item.getString("file"))) {
                    image = BitmapFactory.decodeStream(input);
                }
                assertNotNull("invalid raster", image);
                int w = image.getWidth(), h = image.getHeight();
                int[] pixels = new int[w*h];
                image.getPixels(pixels, 0, w, 0, 0, w, h);
                image.recycle();
                for (String pipeline : new String[]{"umliva-nv21", "feldexa-gray-bitmap"}) {
                    long start = SystemClock.elapsedRealtime();
                    Bitmap gray = null;
                    try {
                        InputImage input;
                        if (pipeline.equals("umliva-nv21")) {
                            input = InputImage.fromByteArray(nv21(pixels,w,h), (w+1)&~1, (h+1)&~1, 0, InputImage.IMAGE_FORMAT_NV21);
                        } else {
                            int[] opaque = new int[pixels.length];
                            for (int j = 0; j < pixels.length; j++) {
                                int c = pixels[j];
                                int value = (77*((c>>16)&255)+150*((c>>8)&255)+29*(c&255)+128)>>8;
                                opaque[j] = 0xff000000 | (value<<16) | (value<<8) | value;
                            }
                            gray = Bitmap.createBitmap(opaque,w,h,Bitmap.Config.ARGB_8888);
                            input = InputImage.fromBitmap(gray,0);
                        }
                        Text result = Tasks.await(recognizer.process(input), 30, TimeUnit.SECONDS);
                        int lines = 0;
                        for (Text.TextBlock block : result.getTextBlocks()) lines += block.getLines().size();
                        rows.put(new JSONObject().put("id", item.getString("id")).put("variant", item.getString("variant"))
                            .put("pipeline", pipeline).put("text", result.getText()).put("line_count", lines)
                            .put("elapsed_ms", SystemClock.elapsedRealtime()-start).put("status", "ok"));
                    } finally { if (gray != null) gray.recycle(); }
                }
            }
        } finally { recognizer.close(); }
        try (OutputStream output = target.openFileOutput("observations.json", Context.MODE_PRIVATE)) {
            output.write(rows.toString().getBytes(StandardCharsets.UTF_8));
        }
    }
}
