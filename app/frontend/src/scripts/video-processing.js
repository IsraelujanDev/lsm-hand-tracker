// public/scripts/video-processor.js
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// Finger connection indices and colors, matching MediaPipe's Python defaults
const FINGER_CONNECTIONS = {
    thumb: [[0, 1], [1, 2], [2, 3], [3, 4]],
    index: [[5, 6], [6, 7], [7, 8]],
    middle: [[9, 10], [10, 11], [11, 12]],
    ring: [[13, 14], [14, 15], [15, 16]],
    pinky: [[17, 18], [18, 19], [19, 20]],
    palm: [[0, 5], [5, 9], [9, 13], [13, 17], [0, 17]]
};
const FINGER_COLORS = {
    thumb: [255, 0, 0],  // red
    index: [0, 255, 0],  // green
    middle: [0, 0, 255],  // blue
    ring: [255, 255, 0],  // yellow
    pinky: [255, 0, 255],  // magenta
    palm: [200, 200, 200]   // light gray
};

let video, overlay, ctx, offscreen, landmarker, latestLandmarks;

// Initialize webcam, model, and canvases
const enableVideoProcessing = async (videoId = 'webcam', overlayId = 'overlay') => {
    video = document.getElementById(videoId);
    overlay = document.getElementById(overlayId);
    ctx = overlay.getContext('2d');
    offscreen = document.createElement('canvas');

    // Setup webcam at highest available resolution
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1920 }, height: { ideal: 1080 } }
        });
        video.srcObject = stream;
        await new Promise(r => video.onloadedmetadata = r);
    } catch {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await new Promise(r => video.onloadedmetadata = r);
    }

    // Match overlay and offscreen canvas size to video
    overlay.width = offscreen.width = video.videoWidth;
    overlay.height = offscreen.height = video.videoHeight;

    // Initialize MediaPipe HandLandmarker in VIDEO mode
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: '/models/hand_landmarker.task',
            delegate: "GPU"
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,

    });

    requestAnimationFrame(processFrame);
};

// Main loop: detect landmarks and draw color-coded skeleton
const processFrame = (timestamp) => {
    const result = landmarker.detectForVideo(video, timestamp);
    latestLandmarks = result.landmarks || [];

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    latestLandmarks.forEach(hand => {
        // Draw each finger with its color
        for (const finger in FINGER_CONNECTIONS) {
            const connections = FINGER_CONNECTIONS[finger];
            const [r, g, b] = FINGER_COLORS[finger];
            ctx.strokeStyle = `rgb(${r},${g},${b})`;
            ctx.lineWidth = 8;
            connections.forEach(([start, end]) => {
                const p1 = hand[start];
                const p2 = hand[end];
                ctx.beginPath();
                ctx.moveTo(p1.x * overlay.width, p1.y * overlay.height);
                ctx.lineTo(p2.x * overlay.width, p2.y * overlay.height);
                ctx.stroke();
            });
        }
        // Draw the landmark circles in white
        ctx.fillStyle = 'white';
        hand.forEach(({ x, y }) => {
            ctx.beginPath();
            ctx.arc(x * overlay.width, y * overlay.height, 8, 0, 2 * Math.PI);
            ctx.fill();
        });
    });

    requestAnimationFrame(processFrame);
};

// Capture a high-quality snapshot and save JSON
const captureSnapshot = (buttonId = 'captureBtn') => {
    document.getElementById(buttonId).addEventListener('click', () => {
        const id = crypto.randomUUID();
        // Raw frame -> PNG
        const offCtx = offscreen.getContext('2d');
        offCtx.drawImage(video, 0, 0);
        offscreen.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a'); a.href = url;
            a.download = `frame_${id}.png`; a.click(); URL.revokeObjectURL(url);
        }, 'image/png');
        // Landmarks -> JSON
        const payload = { id, timestamp: Date.now(), landmarks: latestLandmarks };
        const jb = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const jurl = URL.createObjectURL(jb);
        const ja = document.createElement('a'); ja.href = jurl;
        ja.download = `landmarks_${id}.json`; ja.click(); URL.revokeObjectURL(jurl);
    });
};

export { enableVideoProcessing, captureSnapshot };
