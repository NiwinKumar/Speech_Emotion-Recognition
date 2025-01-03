72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
# Speech Emotion Recognition System
- IS09_emotion: 384 features 
- IS10_paraling: 1582 features 
- IS11_speaker_state: 4368 features 
- IS12_speaker_trait: 6125 features 
- IS13_ComParE: 6373 features 
- ComParE_2016: 6373 features 

### Preprocessing Data

To extract features from your dataset, run:

```bash
python preprocess.py --config configs/example.yaml
```

This command will process audio files and store extracted features in designated formats.

### Training the Model

Ensure your dataset is organized with emotion-specific folders. Then, execute:

```bash
python train.py --config configs/example.yaml
```

### Making Predictions

To predict emotions from a new audio file after training, modify the `predict.py` script with the path to your audio file, then run:

```bash
python predict.py --config configs/example.yaml
```

## Visualization Functions

Several utility functions are provided to visualize results:

### Radar Chart

To display predicted probabilities in a radar chart:

```python
import utils

# Example usage:
data_prob = np.array([0.1, 0.2, 0.3, 0.4]) # Replace with actual probabilities
class_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
utils.radar(data_prob, class_labels)
```

### Loss/Accuracy Curve

To visualize training loss or accuracy over epochs:

```python
import utils

# Example usage:
train_loss = [0.8, 0.6, 0.4] # Replace with actual training loss values
val_loss = [0.9, 0.7, 0.5] # Replace with actual validation loss values
utils.curve(train_loss, val_loss, title='Loss Curve', y_label='Loss')
```

### Waveform Visualization

To plot the waveform of an audio file:

```python
import utils

file_path = 'path/to/audio/file.wav'
utils.waveform(file_path)
```

### Spectrogram Visualization

To plot a spectrogram of an audio file:

```python
import utils

file_path = 'path/to/audio/file.wav'
utils.spectrogram(file_path)
```

## Conclusion

This SER system leverages state-of-the-art deep learning techniques to classify emotions from speech effectively. By utilizing a combination of LSTM, CNN, SVM, and MLP models along with robust feature extraction methods, it aims to enhance applications in areas such as customer service and mental health analysis.

For further details or contributions, please refer to the project repository or contact me directly.
