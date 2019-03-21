# T-shirt-based-Person-Tracking
Developed a real-time system to track the person wearing specific t-shirt in a live streaming video.

This application make use of python and OpenCv Library. Few features of the application includes

1.	Detect a logo T-shirt in the video. It is advisible to use any T-shirt with a logo that is big and easy to recognize.

2.	The task is to detect and track only the person wearing that specific T-shirt based on LogoFinal image, i.e., put a bounding box on the entire person wearing the T-shirt (not just the T-shirt alone).

3.	Additionally, mark the tracked person’s face and eyes with different colored bounding boxes.

4.	A separate task for the project is to determine the height of the person in feet and inches. To do I have used an object of known width and height to act as a calibration parameter. To need to track that object and the person, find the ratio and obtain the person’s real height. In order to do this, the person needs to be standing next to the object, at the same depth location.
