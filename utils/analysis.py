from PIL import Image
import io
import matplotlib.pyplot as plt
from pathlib import Path

#animation methods
def fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib figure to PIL Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)
    return im


def write_mp4(frames, out_fname: Path, fps: int = 4, codec: str = "libx264"):
        """
        Write MP4 using imageio (ffmpeg backend). Frames should be PIL Images.
        """
        import numpy as np
        import imageio.v2 as imageio  # pip install imageio imageio-ffmpeg

        out_fname = Path(out_fname)
        out_fname.parent.mkdir(parents=True, exist_ok=True)

        # Ensure consistent size + even dims for broad MP4 compatibility (yuv420p)
        w, h = frames[0].size
        if (w % 2) == 1: w += 1
        if (h % 2) == 1: h += 1

        writer = imageio.get_writer(
            str(out_fname),
            fps=fps,
            codec=codec,
            pixelformat="yuv420p",
        )

        try:
            for im in frames:
                if im.size != (w, h):
                    im = im.resize((w, h))
                arr = np.asarray(im.convert("RGB"))  # MP4 doesn't support alpha
                writer.append_data(arr)
        finally:
            writer.close()

        return out_fname