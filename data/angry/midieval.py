import midi
import midi.timeresolver as tres
import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mid", type=str, help="the midi file you want to analyze")
args = parser.parse_args()
print(args.mid)

pattern = midi.read_midifile(args.mid)
pattern.make_ticks_abs()
time_resolver = tres.TimeResolver(pattern)
total_ms = 0
for track in pattern:
    for event in track:
        name = event.name
        if "Note" in name:
                pitch = event.pitch
                velocity = event.velocity
        else:
                pitch = 0
                velocity = 0
        tick = event.tick
        milliseconds = time_resolver.tick2ms(tick)
        total_ms += milliseconds
        print(f"event {name} with MIDI tick {tick} and pitch,velocity {pitch},{velocity} happens after {milliseconds} milliseconds.")
print(total_ms)