from abc import ABCMeta, abstractmethod
from typing import List, Tuple
from numbers import Real

import os

from scipy import stats
from numpy.random import Generator
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

class RandomRenderer:
    """Renders text according to random variables with
    pre-specified distributions.
    """
    def __init__(
        self,
        rng: Generator,
        size: Tuple[int, int] = (600, 600),
        top_left: Tuple[int, int] = (20, 20),
        fonts_dir: str = '/usr/share/fonts',
        orientation_dist: Tuple[Real, Real, Real, Real]
            = (0.25, 0.25, 0.25, 0.25),
        fontsize_mean: Real = 14,
        fontsize_std: Real = 3,
        background_color_means: Tuple[Real, Real, Real]
            = (230, 230, 230),
        background_color_stds: Tuple[Real, Real, Real]
            = (20, 20, 20),
        foreground_color_means: Tuple[Real, Real, Real]
            = (20, 20, 20),
        foreground_color_stds: Tuple[Real, Real, Real]
            = (20, 20, 20)
    ):
        self.rng = rng
        self.size = size
        self.top_left = top_left
        assert os.path.exists(fonts_dir), '{} does not exist.'.format(
            fonts_dir)
        self.fonts = get_truetype_fonts(fonts_dir)
        assert len(self.fonts) > 0, ('The directory given by fonts_dir must '
                                     'contain at least one .ttf file, but {}'
                                     'and its subdirectories have not .ttf '
                                     'files'.format(fonts_dir))
        self.orientation_rv = stats.rv_discrete(
            name='orientation',
            values=((0, 90, 270, 180), orientation_dist),
            seed=rng
        )
        self.fontsize_rv = Normal(fontsize_mean, fontsize_std, rng)
        self.background_color_rvs = [
            Normal(mean, std, rng)
            for mean, std in zip(background_color_means, background_color_stds)
        ]
        self.foreground_color_rvs = [
            Normal(mean, std, rng)
            for mean, std in zip(foreground_color_means, foreground_color_stds)
        ]
    def render(self, text: str) -> Image:
        """Renders `text` and returns the resulting image."""
        img = Image.new('RGB', self.size, (
            *[to_color(rv.rvs()) for rv in self.background_color_rvs], 0
        ))
        ImageDraw.Draw(img).text(
            self.top_left,
            text=text,
            font=self._get_font(text),
            fill=tuple(to_color(rv.rvs()) for rv in self.foreground_color_rvs)
        )
        return img.rotate(self.orientation_rv.rvs())
    def _get_font(
        self, text: str,
        required_n_renderable_chars: int = 3
    ) -> ImageFont:
        """Returns a font that is guaranteed to be
        able to render at least `required_n_renderable_chars` characters
        of `text`.
        """
        return ImageFont.truetype(
            self._get_font_path(text, required_n_renderable_chars),
            size=max(0, round(self.fontsize_rv.rvs()))
        )
    def _get_font_path(
        self, text: str,
        required_n_renderable_chars: int = 3
    ) -> str:
        """Returns a path to a TrueType font that is guaranteed to be
        able to render at least `required_n_renderable_chars` characters
        of `text`.
        """
        self.rng.shuffle(self.fonts)
        if len(text) > 0:
            characters = [
                self.rng.choice(list(text))
                for _ in range(required_n_renderable_chars)
            ]
            for font in self.fonts:
                if all(font_has_char(font, c) for c in characters):
                    return font
        return self.fonts[0]

class RV(metaclass=ABCMeta):
    @abstractmethod
    def rvs(self): pass

RV.register(stats.rv_continuous)

class Normal(RV):
    """Represents a normal distribution with mean `mu` and standard
    deviation `std`.
    """
    def __init__(self, mu: Real, std: Real, rng: Generator):
        self.mu = mu
        self.std = std
        self.rng = rng
    def rvs(self):
        return stats.norm.rvs(
            loc=self.mu,
            scale=self.std,
            random_state=self.rng
        )

def to_color(x: Real) -> int:
    """Returns the valid RGB value (i.e., integer in [0, 255]) that is
    closest to `x`.
    """
    return max(0, min(255, int(round(x))))

def font_has_char(font_path: str, character: str) -> bool:
    """Returns whether the TrueType font located at `font_path` has the
    character `character`.
    """
    assert len(character) == 1
    font = TTFont(font_path)
    return any(
        table.isUnicode() and ord(character) in table.cmap
        for table in font['cmap'].tables
    )

def get_truetype_fonts(fonts_dir: str = '/usr/share/fonts') -> List[str]:
    """Returns paths to all TrueType fonts located in `fonts_dir`."""
    return [
        os.path.join(directory, file)
        for (directory, _, files) in os.walk(fonts_dir)
        for file in files
        if file.endswith('.ttf')
    ]