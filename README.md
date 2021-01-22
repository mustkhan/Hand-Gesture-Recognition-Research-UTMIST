<a href="https://aimeos.org/">
    <img src="Images/round-logo.png" alt="UTMIST: Hand Gesture Recognition System" title="UTMIST: Hand Gesture Recognition System" align="right" height="80" />
</a>

# Hand Gesture Recognition Research | UTMIST 👋 

## Introduction

This research work is on the development of a complex hand gesture recognition system that can interface with a website, games and a robot all using the built in webcam of a computer. In this currently on going project, I am responsible for researching contemporary machine learning approaches to achieving this goal, find an appropriate dataset and develop a model and train it to have a high performance accuracy. I am conducting the research work and development with a group of dedicated students in the University Of Toronto Machine Intelligence Student Team.

Currently we are developing the components needed for the final system in a modular fashion and will be integrating these parts by Feb-March of 2021. We aim to submit our insight and work at a conference! 

## Steps To Use:

1. Download best weights: https://drive.google.com/drive/folders/1t4JcH-Y5rIvTWbKiEQ-x2_5NsUg8mP_L?usp=sharing and copy the absolute path of where you placed this.

2. Install requirements.

3. Use: `python webcam.py -e False -u False -cp [insert absolute path of weights here]` to run inferences using your webcam. (If you have a GPU, use -u True) 

4. if you want to train, the paths in the config files will have to be changed so don't worry about modifying those. 

