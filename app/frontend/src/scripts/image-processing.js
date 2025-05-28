import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks(
    '/mediapipe/wasm'
);

const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
        modelAssetPath: '/models/hand_landmarker.task'
    },
    numHands: 2
});