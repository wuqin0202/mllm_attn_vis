import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vlm_attention_viz.render import (
    attention_colors,
    attention_masses,
    normalize_slice,
    overlay_attention,
    render_heatmap,
    render_selection,
    select_attention_slice,
)


class RenderTests(unittest.TestCase):
    def test_shared_normalization_and_zero_map(self):
        visual, context, vmax = attention_colors([0.0, 2.0], [1.0, 4.0])
        np.testing.assert_allclose(visual, [0.0, 0.5])
        np.testing.assert_allclose(context, [0.25, 1.0])
        self.assertEqual(vmax, 4.0)
        zero = render_heatmap([0, 0, 0, 0], (2, 2), (4, 4))
        self.assertTrue(np.all(np.asarray(zero) == 0))

    def test_mean_selection_is_float32_and_grid_is_strict(self):
        layer = SimpleNamespace(
            visual=np.array([[[1, 3], [2, 4]], [[3, 5], [4, 6]]], dtype=np.float16),
            context=np.ones((2, 2, 3), dtype=np.float16),
        )
        visual, context = select_attention_slice(layer, 1, "mean")
        self.assertEqual(visual.dtype, np.float32)
        np.testing.assert_allclose(visual, [3, 5])
        self.assertEqual(context.shape, (3,))
        with self.assertRaises(ValueError):
            render_heatmap([1, 2, 3], (2, 2), (8, 8))

    def test_overlay_alpha_and_nonfinite_rejected(self):
        image = Image.new("RGB", (2, 2), "white")
        result = overlay_attention(image, [0, 1, 0, 0], (2, 2), alpha=0)
        self.assertEqual(result, image)
        result = np.asarray(overlay_attention(image, [0, 1, 0, 0], (2, 2), alpha=1))
        np.testing.assert_array_equal(result[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(result[0, 1], [255, 0, 0])
        with self.assertRaises(ValueError):
            normalize_slice([np.nan], [0])
        with self.assertRaises(ValueError):
            overlay_attention(image, [1, 0, 0, 0], (2, 2), alpha=1.1)

    def test_spatial_overlay_uses_visual_scale_not_context_outlier(self):
        session = SimpleNamespace(
            selectable_positions=np.array([3]),
            generated_positions=np.array([3]),
            layers={
                1: SimpleNamespace(
                    visual=np.array([[[0.001, 0.002]]], dtype=np.float32),
                    context=np.array([[[0.9]]], dtype=np.float32),
                )
            },
            image=Image.new("RGB", (2, 1), "white"),
            visual_grid_hw=(1, 2),
            context_key_positions=np.array([0]),
            tokens=[
                SimpleNamespace(
                    absolute_position=0,
                    token_id=1,
                    raw_piece="text",
                    decoded_preview="text",
                    segment="prompt",
                    is_special=False,
                )
            ],
        )

        rendered = np.asarray(render_selection(session, 3, 1, 0, opacity=1).overlay)

        np.testing.assert_array_equal(rendered[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(rendered[0, 1], [255, 0, 0])

    def test_text_only_selection_has_no_spatial_overlay(self):
        session = SimpleNamespace(
            selectable_positions=np.array([0, 1]),
            layers={
                1: SimpleNamespace(
                    visual=np.empty((1, 2, 0), dtype=np.float32),
                    context=np.array([[[0.0, 0.0], [0.25, 0.0]]], dtype=np.float32),
                )
            },
            image=None,
            visual_grid_hw=None,
            context_key_positions=np.array([0, 1]),
            tokens=[
                SimpleNamespace(
                    absolute_position=position,
                    token_id=position,
                    raw_piece=str(position),
                    decoded_preview=str(position),
                    segment="prompt",
                    is_special=False,
                )
                for position in range(2)
            ],
        )

        rendered = render_selection(session, 1, 1, 0)

        self.assertIsNone(rendered.overlay)
        self.assertEqual(rendered.visual.size, 0)

    def test_raw_masses_preserve_groups(self):
        tokens = [
            SimpleNamespace(absolute_position=0, segment="prompt", is_special=False),
            SimpleNamespace(absolute_position=1, segment="prompt", is_special=True),
            SimpleNamespace(absolute_position=3, segment="generated", is_special=False),
        ]
        self.assertEqual(
            attention_masses([1, 2, 3], [0, 1, 3], tokens),
            {"prompt": 1.0, "generated_history": 3.0, "special_other": 2.0},
        )


if __name__ == "__main__":
    unittest.main()
