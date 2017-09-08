import math
from PIL import Image,ImageDraw

people=['Charlie','Augustus','Veruca','Violet','Mike','Joe','Willy','Miranda']

links=[('Augustus', 'Willy'),
       ('Mike', 'Joe'),
       ('Miranda', 'Mike'),
       ('Violet', 'Augustus'),
       ('Miranda', 'Willy'),
       ('Charlie', 'Mike'),
       ('Veruca', 'Joe'),
       ('Miranda', 'Augustus'),
       ('Willy', 'Augustus'),
       ('Joe', 'Charlie'),
       ('Veruca', 'Augustus'),
       ('Miranda', 'Joe')]

scale = 800
pct = 0.03
domain=[(scale*pct,scale*(1-pct))]*(len(people)*2)

def crosscount(v):
	loc = dict([(people[i], (v[i*2],v[i*2+1])) for i in xrange(len(people))])
	total = 0.0

	for i in xrange(len(links)):
		for j in xrange(i+1, len(links)):

			(x1, y1), (x2, y2) = loc[links[i][0]], loc[links[i][1]]
			(x3, y3), (x4, y4) = loc[links[j][0]], loc[links[j][1]]

			den = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)

			if den == 0: continue

			ua=((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/float(den)
			ub=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/float(den)

			if ua>0 and ua<1 and ub>0 and ub<1: total += 3.2

			costheta = ((x2-x1)*(x4-x3) + (y2-y1)*(y4-y3))/float(math.sqrt(pow((x2-x1),2)+pow((y2-y1),2))*math.sqrt(pow((x4-x3),2)+pow((y4-y3),2)))
			if abs(costheta)>0.5:
				total += 2*abs(costheta)

	for i in range(len(people)):
		for j in range(i+1,len(people)):
        # Get the locations of the two nodes
			(x1,y1),(x2,y2)=loc[people[i]],loc[people[j]]

        # Find the distance between them
			dist=math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
        # Penalize any nodes closer than 50 pixels
		if dist<50:
			total+=(2.0-(dist/50.0))
	return total

def drawnet(sol):

	img = Image.new('RGB', (scale,scale),(255,255,255))
	draw = ImageDraw.Draw(img)

	pos = dict([(people[i], (sol[i*2],sol[i*2+1])) for i in xrange(len(people))])

	for (a,b) in links:
		draw.line((pos[a], pos[b]), fill=(255,0,0))

	for n,p in pos.items():
		draw.text(p, n, (0,0,0))
	# img.show()
	img.save('net.jpg','JPEG')