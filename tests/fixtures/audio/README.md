# Fixed Mandarin audio compression fixtures

`clean.mp3` and `clean.amr` are byte-identical retained derivatives of the public
AISHELL-1 test utterance already used in `packaging/probes/asr.wav`:
`甚至出现交易几乎停滞的情况`.

The pinned dataset revision, public source URL, parent/input/output SHA256,
attribution, compression and resampling details are in `source.json`.
The dataset card declares Apache-2.0; the complete license is included in
`LICENSE.Apache-2.0`. These sample bytes retain their source terms rather than
being relabeled as Tracefolio's AGPL code.

MP3 is the retained 32 kbps derivative. AMR is the retained 8 kHz mono
AMR-NB 12.2 kbps derivative, with final-frame zero padding. They preserve the
original research output and encoder timestamps; no sample was regenerated or
cleaned up to remove the first MP3 clock failure. The fixture check fails if
bytes change or disappear. Neither variant is an independent speech recording
or broad quality benchmark.

`TestRealASRArchiveFormatsClockAndOfflineSearch` uses these committed files so
clean GitHub CI has no dependency on ignored experiment samples. Production
inference uses the frozen portable runtime; fixture creation does not run at
test time. M4A/AAC/FLAC and the 130 second repeated pipeline stress sample are
authored in bounded temporary test storage from the same fixed WAV.
