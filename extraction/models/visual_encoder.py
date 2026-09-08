
import torch
import numpy as np

from PIL import Image


class QwenVisualEncoder:

    def __init__(self, model, layers=None):

        self.model = model

        # Qwen3.5 multimodal visual encoder
        self.visual = self.model.model.visual

        self.features = {}
        self.handles = []

        # If layers=None, capture all vision layers
        if layers is None:
            self.layers = set(
                range(len(self.visual.blocks))
            )
        else:
            self.layers = set(layers)

        self._register_hooks()


    def _register_hooks(self):

        # ----------------------------
        # Vision transformer blocks
        # ----------------------------

        for layer_index, block in enumerate(self.visual.blocks):

            if layer_index not in self.layers:
                continue

            def make_hook(index):

                def hook(module, inputs, output):

                    if isinstance(output, tuple):
                        output = output[0]

                    self.features[f"layer_{index}"] = (
                        output
                        .detach()
                        .cpu()
                    )

                return hook

            handle = block.register_forward_hook(
                make_hook(layer_index)
            )

            self.handles.append(handle)


        # ----------------------------
        # Final merged visual features
        # ----------------------------

        if hasattr(self.visual, "merger"):

            def merger_hook(module, inputs, output):

                if isinstance(output, tuple):
                    output = output[0]

                self.features["merged"] = (
                    output
                    .detach()
                    .cpu()
                )

            handle = self.visual.merger.register_forward_hook(
                merger_hook
            )

            self.handles.append(handle)


    def _prepare_image(self, image):

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif isinstance(image, np.ndarray):

            if image.dtype != np.uint8:

                image = np.clip(
                    image,
                    0,
                    255,
                ).astype(np.uint8)

            image = Image.fromarray(
                image
            ).convert("RGB")

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        else:

            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        return image


    def encode(
        self,
        image,
        processor,
    ):

        self.features.clear()

        image = self._prepare_image(
            image
        )

        # Feed image through the same multimodal
        # pathway used by normal Qwen inference.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": "Describe the visible image.",
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Use the model input embedding device
        input_device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        inputs = {
            key: (
                value.to(input_device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

        # Forward pass triggers the visual hooks
        with torch.inference_mode():

            _ = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

        if len(self.features) == 0:

            raise RuntimeError(
                "No visual features were captured. "
                "The visual module path may not match this Qwen model."
            )

        return {
            key: value.clone()
            for key, value in self.features.items()
        }


    def close(self):

        for handle in self.handles:
            handle.remove()

        self.handles.clear()