# Robot-Localization

This project aim to crate a new robot localization algorithm for systems with low processing capacity with a low measurement rate and a large amount of noise. 
The algorithm has a preference for ease and speed of calculation over accuracy.


This algorithm was tested against a particle filter and an extended Kalman filter. 

In the scenarios tested:

Compared with particle filter: The algorithm runs at 120 times higher speed and 17% lower accuracy .
Compared with EKF: The algorithm runs at 20% lower speed and 17.2 times the accuracy.
