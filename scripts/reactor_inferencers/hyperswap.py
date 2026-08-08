import cv2
import numpy as np
import onnxruntime as ort

from scripts.reactor_logger import logger


class HyperSwapper:
    def __init__(self, model_file, providers=None):
        self.model_file = model_file
        self.providers = providers or ["CPUExecutionProvider"]

        logger.status(
            f"Loading HyperSwap model: {model_file} "
            f"with providers {self.providers}"
        )

        self.session = ort.InferenceSession(
            self.model_file,
            providers=self.providers
        )

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        logger.status(
            "HyperSwap inputs: "
            + ", ".join(
                f"{inp.name}={inp.shape}"
                for inp in self.inputs
            )
        )

    def get_landmarks_5(self, face):
        # InsightFace Face objects normally expose 5-point landmarks as `kps`
        if hasattr(face, "kps") and face.kps is not None:
            return face.kps

        if hasattr(face, "landmark_5") and face.landmark_5 is not None:
            return face.landmark_5

        if hasattr(face, "landmark") and face.landmark is not None:
            if face.landmark.shape[0] >= 68:
                idxs = [36, 45, 30, 48, 54]
                return face.landmark[idxs]

        return None

    def get_affine_transform(self, src_pts, dst_pts):
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        return M

    def create_gradient_mask(self, crop_size=256):
        mask = np.zeros(
            (crop_size, crop_size),
            dtype=np.float32
        )

        center = (
            crop_size // 2,
            crop_size // 2
        )

        axes = (
            int(crop_size * 0.35),
            int(crop_size * 0.40)
        )

        cv2.ellipse(
            mask,
            center,
            axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )

        mask = cv2.GaussianBlur(
            mask,
            (15, 15),
            0
        )

        return np.clip(mask, 0.0, 1.0)

    def paste_back(
        self,
        target_img,
        swapped_face,
        M,
        crop_size=256
    ):
        mask = self.create_gradient_mask(crop_size)
        mask_3c = np.stack([mask] * 3, axis=2)

        h, w = target_img.shape[:2]

        swapped_face_norm = (
            swapped_face.astype(np.float32) / 255.0
        )

        warped_face = cv2.warpAffine(
            swapped_face_norm,
            M,
            (w, h),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.5,
        )

        warped_mask = cv2.warpAffine(
            mask_3c,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        warped_face = np.nan_to_num(
            warped_face,
            nan=0.5,
            posinf=1.0,
            neginf=0.0
        )
        warped_face = np.clip(
            warped_face,
            0.0,
            1.0
        )

        warped_mask = np.nan_to_num(
            warped_mask,
            nan=0.0
        )
        warped_mask = np.clip(
            warped_mask,
            0.0,
            1.0
        )

        warped_mask = cv2.GaussianBlur(
            warped_mask,
            (3, 3),
            0
        )

        target_float = (
            target_img.astype(np.float32) / 255.0
        )

        result_float = (
            target_float * (1.0 - warped_mask)
            + warped_face * warped_mask
        )

        return (
            result_float * 255.0
        ).clip(0, 255).astype(np.uint8)

    def _build_feed(self, source_embedding, target_crop):
        """
        HyperSwap normally names the inputs `source` and `target`.

        This also detects them by tensor rank so we're not completely
        dependent on exact ONNX input names.
        """
        feed = {}

        for inp in self.inputs:
            shape = inp.shape

            # ArcFace identity embedding: normally [1, 512]
            if len(shape) == 2:
                feed[inp.name] = source_embedding

            # Image tensor: normally [1, 3, 256, 256]
            elif len(shape) == 4:
                feed[inp.name] = target_crop

        if len(feed) != 2:
            # Known FaceFusion / HyperSwap naming
            input_names = [x.name for x in self.inputs]

            if "source" in input_names and "target" in input_names:
                feed = {
                    "source": source_embedding,
                    "target": target_crop,
                }
            else:
                raise RuntimeError(
                    "Unable to identify HyperSwap ONNX inputs: "
                    + ", ".join(
                        f"{x.name}={x.shape}"
                        for x in self.inputs
                    )
                )

        return feed

    def get(
        self,
        img,
        target_face,
        source_face,
        paste_back=True
    ):
        # HyperSwap takes the normalized ArcFace embedding directly.
        source_embedding = source_face.normed_embedding

        if source_embedding is None:
            logger.error(
                "HyperSwap: source face has no embedding"
            )
            return img if paste_back else (None, None)

        source_embedding = (
            source_embedding
            .reshape(1, -1)
            .astype(np.float32)
        )

        target_landmarks = self.get_landmarks_5(
            target_face
        )

        if target_landmarks is None:
            logger.error(
                "HyperSwap: target face has no 5-point landmarks"
            )
            return img if paste_back else (None, None)

        # FFHQ-style 256x256 alignment used by
        # ComfyUI-ReActor's HyperSwap implementation.
        std_landmarks_256 = np.array(
            [
                [84.87, 105.94],
                [171.13, 105.94],
                [128.00, 146.66],
                [96.95, 188.64],
                [159.05, 188.64],
            ],
            dtype=np.float32,
        )

        M = self.get_affine_transform(
            target_landmarks.astype(np.float32),
            std_landmarks_256
        )

        if M is None:
            logger.error(
                "HyperSwap: failed to calculate face transform"
            )
            return img if paste_back else (None, None)

        crop = cv2.warpAffine(
            img,
            M,
            (256, 256),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        # BGR -> RGB
        crop_input = (
            crop[:, :, ::-1]
            .astype(np.float32)
            / 255.0
        )

        # [0,1] -> [-1,1]
        crop_input = (
            crop_input - 0.5
        ) / 0.5

        # HWC -> NCHW
        crop_input = (
            crop_input
            .transpose(2, 0, 1)[np.newaxis, ...]
            .astype(np.float32)
        )

        feed = self._build_feed(
            source_embedding,
            crop_input
        )

        try:
            output = self.session.run(
                None,
                feed
            )[0][0]
        except Exception as e:
            logger.error(
                f"HyperSwap inference failed: {e}"
            )
            raise

        output = np.nan_to_num(
            output,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        # HyperSwap usually returns [-1, 1].
        if output.min() < 0.0 or output.max() <= 1.5:
            output = (
                (output + 1.0)
                / 2.0
                * 255.0
            )

        output = np.clip(
            output,
            0,
            255
        ).astype(np.uint8)

        # CHW -> HWC
        output = output.transpose(
            1,
            2,
            0
        )

        # RGB -> BGR, since ReActor's working image is BGR.
        output = output[:, :, ::-1]

        if not paste_back:
            return output, M

        return self.paste_back(
            img,
            output,
            M,
            crop_size=256
        )
