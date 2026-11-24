package org.mindloop.mlkitlab;

import android.graphics.Rect;
import org.json.JSONArray;
import org.json.JSONObject;

/** Deterministic, ground-truth-free, one-region policy. No private data or expected labels. */
final class RegionPolicy {
    private RegionPolicy() {}

    static Rect select(JSONArray lines, int width, int height) throws Exception {
        return select(lines, width, height, "region-v1");
    }

    static Rect select(JSONArray lines, int width, int height, String policy) throws Exception {
        if (!policy.equals("region-v1") && !policy.equals("region-v2-small") && !policy.equals("region-v3-short"))
            throw new IllegalArgumentException("Unknown policy");
        int[] heights = new int[lines.length()];
        for (int i = 0; i < lines.length(); i++) {
            JSONArray b = lines.getJSONObject(i).getJSONArray("box");
            if (b.length() != 4 || b.getInt(3) <= b.getInt(1))
                throw new IllegalArgumentException("Bad line geometry");
            heights[i] = b.getInt(3) - b.getInt(1);
        }
        java.util.Arrays.sort(heights);
        int median = heights.length == 0 ? 0 : heights[heights.length / 2];
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Bad raster");
        Rect best = null;
        for (int i = 0; i < lines.length(); i++) {
            JSONObject item = lines.getJSONObject(i);
            String text = item.getString("text").trim();
            int count = text.codePointCount(0, text.length());
            boolean shortOrDamaged = count <= (policy.equals("region-v3-short") ? 5 : 2) || text.indexOf('\uFFFD') >= 0;
            if (!policy.equals("region-v2-small") && !shortOrDamaged) continue;
            JSONArray b = item.getJSONArray("box");
            if (b.length() != 4) throw new IllegalArgumentException("Bad line geometry");
            Rect line = new Rect(b.getInt(0), b.getInt(1), b.getInt(2), b.getInt(3));
            if (line.width() <= 0 || line.height() <= 0) throw new IllegalArgumentException("Bad line bounds");
            if (policy.equals("region-v2-small") &&
                    (heights.length < 3 || 10L * line.height() > 7L * median)) continue;
            int padX = Math.max(16, Math.min(120, line.height() * 2));
            int padY = Math.max(12, Math.min(80, line.height()));
            Rect region = new Rect(Math.max(0, line.left - padX),
                    Math.max(0, line.top - padY),
                    Math.min(width, line.right + padX),
                    Math.min(height, line.bottom + padY));
            if (region.isEmpty() || (long) region.width() * region.height() > (long) width * height / 2) continue;
            if (best == null || region.top < best.top || (region.top == best.top && region.left < best.left)) best = region;
        }
        return best;
    }

    static Rect toSource(Rect crop, Rect recognized, int scale, int width, int height) {
        if (scale < 1 || crop.isEmpty() || recognized.isEmpty()) throw new IllegalArgumentException("Invalid crop");
        int left = Math.max(0, Math.min(width, crop.left + recognized.left / scale));
        int top = Math.max(0, Math.min(height, crop.top + recognized.top / scale));
        int right = Math.max(0, Math.min(width, crop.left + (recognized.right + scale - 1) / scale));
        int bottom = Math.max(0, Math.min(height, crop.top + (recognized.bottom + scale - 1) / scale));
        if (right <= left || bottom <= top) throw new IllegalArgumentException("Empty mapped line");
        return new Rect(left, top, right, bottom);
    }
}
