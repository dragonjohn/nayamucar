from common.stuff  import *
from common.pid    import PID
from common        import imgutils

import math


@AutoPilot.register
class TrendCarPilot(AutoPilot):
    _track_view_range = (0.6, 0.85)


    def __init__(self):
        pass


    @AutoPilot.priority_normal
    def on_inquiry_drive(self, dashboard, last_result):
        if not self.get_autodrive_started():
            return None

        steering = self._find_steering_angle_by_color(dashboard)
        if steering < -90:
            return {"steering": 0.0, "throttle": -0.2}

        throttle = 0.7 - min(abs(steering / 50.0), 0.5)
        return {"steering": steering, "throttle": throttle}


    def _find_steering_angle_by_color(self, dashboard):
        if "frame" not in dashboard:
            return -100.0   # special case

        frame             = dashboard["frame"]
        img_height        = frame.shape[0]
        img_width         = frame.shape[1]
        camera_x          = img_width // 2

        track_view_slice  = slice(*(int(x * img_height) for x in self._track_view_range))
        track_view        = self._flatten_rgb(frame[track_view_slice, :, :])

        track_view_gray   = cv2.cvtColor(track_view, cv2.COLOR_BGR2GRAY)
        tracks            = map(lambda x: len(x[x > 20]), [track_view_gray])
        tracks_seen       = filter(lambda y: y > 2000, tracks)

        if len(list(tracks_seen)) == 0:
            show_image("frame", frame)             # display image to opencv window
            show_image("track_view", track_view)   # display image to opencv window

            # show track image to webconsole
            dashboard["track_view"     ] = track_view
            dashboard["track_view_info"] = (track_view_slice.start, track_view_slice.stop, None)
            return -100.0   # special case

        _y, _x  = np.where(track_view_gray == 76)

        px = np.mean(_x)
        if np.isnan(px):
            show_image("frame", frame)             # display image to opencv window
            show_image("track_view", track_view)   # display image to opencv window

            # show track image to webconsole
            dashboard["track_view"     ] = track_view
            dashboard["track_view_info"] = (track_view_slice.start, track_view_slice.stop, None)
            return -100.0   # special case

        steering_angle = math.atan2(track_view.shape[0] * float(2.5), (px - camera_x))
        #steering_angle = math.atan2(img_height, (px - camera_x))

        #draw the steering direction and display on webconsole
        r = 60
        x = track_view.shape[1] // 2 + int(r * math.cos(steering_angle))
        y = track_view.shape[0]      - int(r * math.sin(steering_angle))
        cv2.line(track_view, (track_view.shape[1] // 2, track_view.shape[0]), (x, y), (255, 0, 255), 2)

        show_image("frame", frame)             # display image to opencv window
        show_image("track_view", track_view)   # display image to opencv window

        # show track image to webconsole
        dashboard["track_view"     ] = track_view
        dashboard["track_view_info"] = (track_view_slice.start, track_view_slice.stop, (np.pi/2 - steering_angle) * 180.0 / np.pi)
        return (np.pi/2 - steering_angle) * 180.0 / np.pi


    def _flatten_rgb(self, img):
        b, g, r = cv2.split(img)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        y_filter = ((b >= 128) & (g >= 128) & (r < 100))

        b[y_filter], g[y_filter] = 255, 255
        r[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((b, g, r))
        return flattened

