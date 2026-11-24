package org.mindloop.mlkitlab;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
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
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ResearchTest {
    static class Region {
        final int left, top, right, bottom;
        Region(int l, int t, int r, int b) { left=l; top=t; right=r; bottom=b; }
        int width() { return right-left; }
        int height() { return bottom-top; }
        long area() { return (long)width()*height(); }
    }

    static int clamp(int value) { return Math.max(0, Math.min(255, value)); }

    static String normalized(String text) {
        return Normalizer.normalize(text, Normalizer.Form.NFC).trim().replaceAll("\\s+", " ");
    }

    static boolean overlaps(Region a, Region b) {
        int w=Math.max(0, Math.min(a.right,b.right)-Math.max(a.left,b.left));
        int h=Math.max(0, Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top));
        return (long)w*h >= 0.15*Math.min(a.area(),b.area());
    }

    static List<Region> retryRegions(JSONArray lines, int width, int height) throws Exception {
        List<JSONObject> ordered = new ArrayList<>();
        for (int i=0;i<lines.length();i++) ordered.add(lines.getJSONObject(i));
        ordered.sort(Comparator.comparingInt((JSONObject row)->row.optJSONArray("box").optInt(1))
                .thenComparingInt(row->row.optJSONArray("box").optInt(0)));
        List<Region> regions = new ArrayList<>();
        for (JSONObject line: ordered) {
            String text=normalized(line.getString("text"));
            int count=text.codePointCount(0,text.length());
            if (count>2 && !text.contains("\ufffd")) continue;
            JSONArray box=line.getJSONArray("box");
            int x1=box.getInt(0), y1=box.getInt(1), x2=box.getInt(2), y2=box.getInt(3);
            if (x2<=x1 || y2<=y1) throw new IllegalArgumentException("Invalid OCR box");
            int lineHeight=y2-y1, padX=Math.max(24,lineHeight), padY=Math.max(12,lineHeight/2);
            Region region=new Region(Math.max(0,x1-padX),Math.max(0,y1-padY),
                    Math.min(width,x2+padX),Math.min(height,y2+padY));
            if (region.width()<=0 || region.height()<=0) throw new IllegalArgumentException("Retry box outside raster");
            boolean duplicate=false;
            for (Region old: regions) if (overlaps(old,region)) { duplicate=true; break; }
            if (!duplicate) regions.add(region);
            if (regions.size()==4) break;
        }
        return regions;
    }

    static JSONArray offsetLines(JSONArray lines, int dx, int dy, int cropWidth, int cropHeight) throws Exception {
        JSONArray out=new JSONArray();
        for (int i=0;i<lines.length();i++) {
            JSONObject line=lines.getJSONObject(i);
            JSONArray b=line.getJSONArray("box");
            int left=Math.max(0,Math.min(cropWidth,b.getInt(0)));
            int top=Math.max(0,Math.min(cropHeight,b.getInt(1)));
            int right=Math.max(0,Math.min(cropWidth,b.getInt(2)));
            int bottom=Math.max(0,Math.min(cropHeight,b.getInt(3)));
            if (right<=left || bottom<=top) continue;
            out.put(new JSONObject().put("text",line.getString("text"))
                    .put("box",new JSONArray(new int[]{left+dx,top+dy,right+dx,bottom+dy})));
        }
        return out;
    }

    static JSONObject recognize(Bitmap image, String pipeline, TextRecognizer recognizer) throws Exception {
        int w=image.getWidth(), h=image.getHeight();
        int[] pixels=new int[w*h];
        image.getPixels(pixels,0,w,0,0,w,h);
        Bitmap gray=null;
        InputImage input;
        if (pipeline.equals("umliva-nv21")) {
            input=InputImage.fromByteArray(nv21(pixels,w,h),(w+1)&~1,(h+1)&~1,0,InputImage.IMAGE_FORMAT_NV21);
        } else {
            int[] opaque=new int[pixels.length];
            for (int j=0;j<pixels.length;j++) {
                int c=pixels[j];
                int value=(77*((c>>16)&255)+150*((c>>8)&255)+29*(c&255)+128)>>8;
                opaque[j]=0xff000000|(value<<16)|(value<<8)|value;
            }
            gray=Bitmap.createBitmap(opaque,w,h,Bitmap.Config.ARGB_8888);
            input=InputImage.fromBitmap(gray,0);
        }
        try {
            Text result=Tasks.await(recognizer.process(input),30,TimeUnit.SECONDS);
            JSONArray lines=new JSONArray();
            for (Text.TextBlock block: result.getTextBlocks()) for (Text.Line line:block.getLines()) {
                Rect box=line.getBoundingBox();
                assertNotNull("missing line bounds",box);
                if (!line.getText().trim().isEmpty()) lines.put(new JSONObject().put("text",line.getText())
                        .put("box",new JSONArray(new int[]{box.left,box.top,box.right,box.bottom})));
            }
            return new JSONObject().put("text",result.getText()).put("lines",lines);
        } finally { if (gray!=null) gray.recycle(); }
    }

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
        Map<String,JSONObject> baselineByKey = new HashMap<>();
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
                for (String pipeline : new String[]{"umliva-nv21", "feldexa-gray-bitmap"}) {
                    long start = SystemClock.elapsedRealtime();
                    String id=item.getString("id"), variant=item.getString("variant");
                    JSONObject row;
                    if (variant.equals("baseline")) {
                        JSONObject result=recognize(image,pipeline,recognizer);
                        JSONArray lines=result.getJSONArray("lines");
                        Runtime memory=Runtime.getRuntime();
                        row=new JSONObject().put("id",id).put("variant",variant).put("pipeline",pipeline)
                                .put("text",result.getString("text")).put("lines",lines).put("line_count",lines.length())
                                .put("heap_used_bytes",memory.totalMemory()-memory.freeMemory())
                                .put("elapsed_ms",SystemClock.elapsedRealtime()-start).put("status","ok");
                        baselineByKey.put(id+"|"+pipeline,row);
                    } else {
                        JSONObject baseline=baselineByKey.get(id+"|"+pipeline);
                        assertNotNull("candidate must follow its baseline",baseline);
                        List<Region> regions=retryRegions(baseline.getJSONArray("lines"),w,h);
                        JSONArray retryLines=new JSONArray();
                        List<String> retryText=new ArrayList<>();
                        for (Region region:regions) {
                            Bitmap crop=Bitmap.createBitmap(image,region.left,region.top,region.width(),region.height());
                            try {
                                JSONObject retry=recognize(crop,pipeline,recognizer);
                                JSONArray mapped=offsetLines(retry.getJSONArray("lines"),region.left,region.top,
                                        region.width(),region.height());
                                for (int k=0;k<mapped.length();k++) retryLines.put(mapped.getJSONObject(k));
                                if (!retry.getString("text").trim().isEmpty()) retryText.add(retry.getString("text").trim());
                            } finally { crop.recycle(); }
                        }
                        Runtime memory=Runtime.getRuntime();
                        row=new JSONObject().put("id",id).put("variant",variant).put("pipeline",pipeline)
                                .put("text",String.join("\n",retryText)).put("lines",retryLines).put("line_count",retryLines.length())
                                .put("retry_regions",regions.size()).put("raster_width",w).put("raster_height",h)
                                .put("heap_used_bytes",memory.totalMemory()-memory.freeMemory())
                                .put("elapsed_ms",SystemClock.elapsedRealtime()-start).put("status","ok");
                    }
                    rows.put(row);
                }
                image.recycle();
            }
        } finally { recognizer.close(); }
        try (OutputStream output = target.openFileOutput("observations.json", Context.MODE_PRIVATE)) {
            output.write(rows.toString().getBytes(StandardCharsets.UTF_8));
        }
    }

    @Test public void regionPolicyClipsAndMapsBoxesBackToSource() throws Exception {
        JSONArray lines=new JSONArray().put(new JSONObject().put("text","?").put("box",new JSONArray(new int[]{1,1,11,21})))
                .put(new JSONObject().put("text","X").put("box",new JSONArray(new int[]{2,2,12,22})))
                .put(new JSONObject().put("text","Müller").put("box",new JSONArray(new int[]{30,50,100,70})))
                .put(new JSONObject().put("text","ß").put("box",new JSONArray(new int[]{200,200,210,220})));
        List<Region> selected=retryRegions(lines,220,215);
        assertEquals(2,selected.size());
        assertEquals(35,selected.get(0).right);
        assertEquals(33,selected.get(0).bottom);
        assertEquals(215,selected.get(1).bottom);
        JSONArray mapped=offsetLines(new JSONArray().put(new JSONObject().put("text","Datum")
                .put("box",new JSONArray(new int[]{2,3,52,44}))),100,200,50,40);
        assertArrayEquals(new int[]{102,203,150,240},new int[]{mapped.getJSONObject(0).getJSONArray("box").getInt(0),
                mapped.getJSONObject(0).getJSONArray("box").getInt(1),mapped.getJSONObject(0).getJSONArray("box").getInt(2),
                mapped.getJSONObject(0).getJSONArray("box").getInt(3)});
    }
}
