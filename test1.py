from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random
fontBase = ["DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf", "DejaVuSansCondensed-Oblique.ttf", "DejaVuSansCondensed-Bold.ttf", "DejaVuSans.ttf", "DejaVuSerifCondensed-BoldItalic.ttf", "DejaVuSerifCondensed-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSans-Bold.ttf",
            "DejaVuSerifCondensed.ttf", "DejaVuSerifCondensed-Italic.ttf", "DejaVuSansCondensed.ttf", "DejaVuSerif-BoldItalic.ttf", "DejaVuSans-ExtraLight.ttf", "DejaVuSansCondensed-BoldOblique.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf"]


def rndChar():
    return chr(random.randint(65, 90))


def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))


def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))


def rndFont():
    return ImageFont.truetype(fontBase[random.randint(0, len(fontBase) - 1)], 36)

width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# font = ImageFont.truetype('DejaVuSerif-Bold.ttf', 36)
draw = ImageDraw.Draw(image)
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=rndFont(), fill=rndColor2(), anchor=30)
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg')
