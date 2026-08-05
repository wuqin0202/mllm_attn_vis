import unittest
from unittest.mock import Mock, patch

from vlm_attention_viz import __main__


class CliTest(unittest.TestCase):
    def test_help_does_not_load_model(self):
        with patch.object(__main__, "load_model") as load_model, \
                self.assertRaises(SystemExit) as raised:
            __main__.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        load_model.assert_not_called()

    def test_launch_wires_model_app_queue_and_server_options(self):
        model = object()
        processor = object()
        queued = Mock()
        demo = Mock()
        demo.queue.return_value = queued

        with patch.object(
            __main__, "load_model", return_value=(model, processor)
        ) as load_model, patch.object(__main__, "build_app", return_value=demo) as build_app:
            __main__.main(
                [
                    "--model",
                    "local/Qwen",
                    "--device",
                    "cuda:3",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8899",
                    "--image-root",
                    "/srv/images",
                    "--image-root",
                    "/mnt/shared/images",
                ]
            )

        load_model.assert_called_once_with("local/Qwen", device="cuda:3")
        build_app.assert_called_once_with(
            model,
            processor,
            device="cuda:3",
            image_roots=["/srv/images", "/mnt/shared/images"],
        )
        demo.queue.assert_called_once_with(default_concurrency_limit=1)
        queued.launch.assert_called_once_with(server_name="127.0.0.1", server_port=8899)


if __name__ == "__main__":
    unittest.main()
