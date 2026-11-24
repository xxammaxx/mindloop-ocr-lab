package org.mindloop.mlkitlab;

import android.graphics.Rect;
import org.json.JSONArray;
import org.json.JSONObject;

/** Deterministic, ground-truth-free, one-region policy. No private data or expected labels. */
final class RegionPolicy {
    private RegionPolicy() {}

    static Rect select(JSONArray lines, int width, int height) throws Exception {
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Bad raster");
        Rect best = null;
        for (int i = 0; i < lines.length(); i++) {
            JSONObject item = lines.getJSONObject(i);
            String text = item.getString("text").trim();
            if (!(text.codePointCount(0, text.length()) <= 2 || text.indexOf('\uFFFD') >= 0)) continue;
            JSONArray b = item.getJSONArray("box");
            if (b.length() != 4) throw new IllegalArgumentException("Bad line geometry");
            Rect line = new Rect(b.getInt(0), b.getInt(1), b.getInt(2), b.getInt(3));
            if (line.width() <= 0 || line.height() <= 0) throw new IllegalArgumentException("Bad line bounds");
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
