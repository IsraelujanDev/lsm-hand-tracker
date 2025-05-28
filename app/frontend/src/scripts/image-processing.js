import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

let handLandmarker = null;

// 1) Initialize the Wasm runtime + model (once)
async function init () {
    if (handLandmarker) return;
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: '/models/hand_landmarker.task'
        },
        runningMode: 'IMAGE',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
}

// 2) Helper to load an image as a bitmap (not via `<img>`)
async function loadImageBitmap (path) {
    const resp = await fetch(path);
    const blob = await resp.blob();
    return await createImageBitmap(blob);
}

/**
 * Public API: load the image at `imagePath` and return its landmarks.
 * @param {string} imagePath  e.g. '/assets/2000_5eb5bec257e4a.png'
 * @returns {Promise<Array<Array<{x:number,y:number,z:number}>>>}
 */
export async function detectLandmarks (imagePath) {
    await init();
    const bitmap = await loadImageBitmap(imagePath);
    const result = handLandmarker.detect(bitmap);
    return result.landmarks;  // Array of hands → each is 21 {x,y,z}
}

// Optional: auto-run on import for testing
detectLandmarks('/public/images/A_fa5d0696.png')
    .then(landmarks => console.log('Landmarks:', landmarks))
    .catch(console.error);
