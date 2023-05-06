class BoundingBox:
    def __init__(self,
    x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @classmethod
    def from_yolo(cls, yolo_box):
        return cls(
          x1=yolo_box.xyxy[0][0].int(),
          y1=yolo_box.xyxy[0][1].int(),
          x2=yolo_box.xyxy[0][2].int(),
          y2=yolo_box.xyxy[0][3].int()
        )

    @classmethod
    def from_xywh(cls, x, y, w, h):
        return cls(
          x1=x,
          y1=y,
          x2=x+w,
          y2=y+h
        )

    @property
    def x(self):
        return self.x1

    @property
    def y(self):
        return self.y1

    # Extracts a portion of an image
    def crop(self, image):
        return image.crop((int(self.x1), int(self.y1), int(self.x2), int(self.y2)))

    def draw(self, draw):
        coords = ((self.x1, self.y1), (self.x2, self.y2))
        draw.rectangle(coords, outline='white', width=2)

    def move(self, x, y):
        return BoundingBox(
            self.x1 + x,
            self.y1 + y,
            self.x2 + x,
            self.y2 + y
        )

    def __repr__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"
