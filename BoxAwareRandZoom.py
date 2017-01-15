import cv2
import random
import numpy as np

def randZoom(img, boxes, minImageRatio=0.3, minBoxRatio=0.8):
	h=img.shape[0]
	w=img.shape[1]

	def sampleStaringPoint(dim, boxPos, boxDim):
		maxVal = np.min([
			int(dim *(1.0 - minImageRatio)),
			int(boxPos+(1.0 - minBoxRatio)*boxDim)
		])

		return random.randint(0, maxVal)

	def sampleEndPoint(startPoint, dim, boxPos, boxDim):
		minBoxSize = int(boxDim * minBoxRatio)
		minImageSize = int(dim * minImageRatio)
		minEndPos = np.max([
			startPoint + minBoxSize + (0 if startPoint >= boxPos else boxPos - startPoint),
			startPoint + minImageSize
		])

		return minEndPos + random.randint(0, dim - minEndPos)


	def growBox(left, top, right, bottom):
		newW = right - left + 1
		newH = bottom - top + 1

		aspectRatio = w/h

		if int(newH*aspectRatio) > newW:
			#Need to grow in X direction
			newW2 = int(newH*aspectRatio)
			assert(newW2 > newW)

			minLeft = np.max([right - newW2, 0])
			maxDiff = np.min([
				left - minLeft,
				w - right
			])

			left = minLeft + random.randint(0, maxDiff)
			right = left + newW2

		elif int(newW/aspectRatio) >= newH:
			#Need to grow in Y direction
			newH2 = int(newW/aspectRatio)
			
			assert(newH2 >= newH)
			minTop = np.max([bottom - newH2, 0])

			maxDiff = np.min([
				top - minTop,
				h - bottom
			])
			
			top = minTop + random.randint(0, maxDiff)
			bottom = top + newH2

		return left, top, right, bottom

	def sampleNoBox():
		left = random.randint(0, int((1.0-minImageRatio)*w))
		right = w-random.randint(0, int(w - left - minImageRatio*w))

		top = random.randint(0, int((1.0-minImageRatio)*h))
		bottom = h-random.randint(0, int(h - top - minImageRatio*h))

		return left, top, right, bottom

	def limitBoxSize(box):
		x=np.min([np.max([box["x"], 0]),w-2])
		y=np.min([np.max([box["y"], 0]),h-2])

		bw = np.min([box["w"] + np.min([box["x"],0]), w-x])
		bw = np.max([bw, 1])
		bh = np.min([box["h"] + np.min([box["y"],0]), h-y])
		bh = np.max([bh, 1])
		return {
			"x": x,
			"y": y,
			"w": bw,
			"h": bh
		}

	def checkBox(left, top, right, bottom, box):
		if box is None:
			return

		cL = np.max([left, box["x"]])
		cR = np.min([right, box["x"]+box["w"]])

		cT = np.max([top, box["y"]])
		cB = np.min([bottom, box["y"]+box["h"]])

		if (cR - cL) < int(minBoxRatio*box["w"]) or (cB - cT) < int(minBoxRatio*box["h"]):
			print("box:",box)
			print("Remaining W: ", cR - cL, "minW:", minBoxRatio*box["w"])
			print("Remaining H: ", cB - cT, "minH:", minBoxRatio*box["h"])
			assert(False)
		pass

	if len(boxes)>1:
		box = boxes[random.randint(0, len(boxes)-1)]
	elif len(boxes)==1:
		box = boxes[0]
	else:
		box = None

	if box is not None:
		box = limitBoxSize(box)
		left = sampleStaringPoint(w, box["x"], box["w"])
		top = sampleStaringPoint(h, box["y"], box["h"])

		right = sampleEndPoint(left, w, box["x"], box["w"])
		bottom = sampleEndPoint(top, h, box["y"], box["h"])
	else:
		left, top, right, bottom = sampleNoBox()

	#checkBox(left, top, right, bottom, box)

	left, top, right, bottom = growBox(left, top, right, bottom)
	
	#checkBox(left, top, right, bottom, box)

	img = img[top:bottom, left:right]
	img = cv2.resize(img, (w,h), cv2.INTER_CUBIC)
	return img	