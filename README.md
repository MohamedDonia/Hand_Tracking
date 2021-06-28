# Hand Tracking
The ability to perceive the shape and motion of hands can be a vital component in improving the user experience across a variety of technological domains and platforms.
For example, it can form the basis for sign language understanding and hand gesture control, and can also enable the overlay of digital content
and information on top of the physical world in augmented reality. While coming naturally to people, robust real-time hand perception is a decidedly
challenging computer vision task, as hands often occlude themselves or each other (e.g. finger/palm occlusions and hand shakes) and lack high contrast patterns.

MediaPipe Hands is a high-fidelity hand and finger tracking solution. 
It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame. Whereas current state-of-the-art
approaches rely primarily on powerful desktop environments for inference, our method achieves real-time performance on a 
mobile phone, and even scales to multiple hands. We hope that providing this hand perception functionality to the wider research and development 
community will result in an emergence of creative use cases, stimulating new applications and new research avenues.

# Hand Landmark Model
After the palm detection over the whole image our subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates
inside the detected hand regions via regression, that is direct coordinate prediction. The model learns a consistent internal hand pose
representation and is robust even to partially visible hands and self-occlusions.

To obtain ground truth data, we have manually annotated ~30K real-world images with 21 3D coordinates, as shown below
(we take Z-value from image depth map, if it exists per corresponding coordinate). To better cover the possible hand poses and provide
additional supervision on the nature of hand geometry, we also render a high-quality synthetic hand model over various backgrounds and map it to the corresponding 3D coordinates.


![handmarks](https://user-images.githubusercontent.com/81274360/123642095-346dd400-d823-11eb-993a-8d230d114eeb.PNG)

