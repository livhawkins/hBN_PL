**Plan for hBN PL project**
- each emitter has a couple of frames
- saving spectra bundles all frames of one emitter into one file

Doing it manually
1. spectralcleaning.py - clean data, removing cosmic rays and really low intensity or highly noisy frames
2. averaging all the frames
3. good emitters fine
4. bad emitters: look for patterns between frames, see how intensity changes

Code pipeline
- identify no of ZPLs, prominence, track over the frames in time, looking at correlations between frames
- identify if triplet/doublet patterns, then isolate spectra to doublet/triplet folder, track ΔI, Δλ over the different frames
- rogue cosmic ray anomalies, anywhere in spectra
- decisions condition
- peak fitter? threshold?
- cleaning the good spectra (of cosmic rays)


- peak fitter: then how many peaks have count > 0.7 (eg)

Conditions for good spectra:
- one signle strong, sharp ZPL, in range 620-630nm
- usually characteristics very similar between frames

Condiitons for bad spectra: 
- noisy
- no clear ZPL
- double ZPL (likely also double PSB)

