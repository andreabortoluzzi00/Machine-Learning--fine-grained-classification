food_template = [
    lambda c: f'a bad photo of a {c.split("_")}.',
    lambda c: f'a slice of {c.split("_")}.',
    lambda c: f'a bowl of {c.split("_")}.',
    lambda c: f'a photo of a piece of {c.split("_")}.',
    lambda c: f'a low resolution photo of some {c.split("_")}.',
    lambda c: f'a plate of {c.split("_")}.',
    lambda c: f'a bad photo of the {c.split("_")}.',
    lambda c: f'a cropped photo of a {c.split("_")}.',
    lambda c: f'a photo of a plate of {c.split("_")} with a fork.',
    lambda c: f'a bright photo of a {c.split("_")}.',
    lambda c: f'a photo of a {c.split("_")}.',
    lambda c: f'a photo of some {c.split("_")}.']