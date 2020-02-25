import numpy as np
import pretty_midi
import os
def piano_roll_to_pretty_midi(piano_roll, fs=1, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def decode(test):
	test = test[:-1]  # takes in the endoded data and looks at all lines except the last line (always blank)
	output = test.split("\n")  # splits on newline character
	res = len(output)  # this is how many samples long the song will be
	arr = np.zeros((128, res))  # initialising a piano roll array with zeros for this length
	timeindex = 0
	for x in output:
		newx = x.replace(")(", "-")  # this is the seperator between different note/velocity pairs in the same sample
		newx = newx.replace("(", "")  # removing the outer bracket
		newx = newx.replace(")", "")  # removing the other outer bracket
		noteGroup = newx.split("-")  # splitting into separate note/velocity pairs
		for n in noteGroup:
			if n != "#":
				s = len(n.split(","))  # this is checking to make sure there aren't more commas then expected
				if s == 2:  # if there is only one  comma then note/velocity follows expected format
					note, velocity = n.split(",")  # splitting into note/velocity
					if velocity.count("#") > 0 or velocity.count('.') > 1:  # these are flaws in generation so replace with 0
						velocity = 0
						note = 0
					elif note.count("#") > 0 or note.count('.') > 1:  # these are flaws in generation so replace with 0
						note = 0
						velocity = 0
					try:
						note = float(note)  # checking for an error
					except:
						note = 0

				if int(note) > 126:  # casting into int if over 126 it isnt a valid midi note so replace with 0
					note = 0
					velocity = 0
				if velocity == "":  # checking for a common error when encoding velocity if found replace with 0
					velocity = 0
					note = 0
				if int(float(velocity)) > 126:  # casting to int if over 126 it isnt a valid midi vel so replace with 0
					velocity = 0
					note = 0
				arr[int(note), timeindex] = velocity  # updating point in piano roll with velocity
		timeindex = timeindex + 1  # incrementing through to next time index (sample) in song
	# arr=arr.T
	return arr


with open('generated.txt') as f:
	lines = f.readlines()
outString=""
for l in lines:
	if len(l.strip()) != 0:
		outString=outString+l

print(outString)
decoded=decode(outString)
fs=20
pm = piano_roll_to_pretty_midi(decoded, fs=fs, program=2)
pm.write("piano.midi")