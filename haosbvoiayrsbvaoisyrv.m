clear

fullPathH51 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-000/wehr3133_am_tuning_curve_20241212a.h5';
fullPathH52 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-002/wehr3133_am_tuning_curve_20241212b.h5';
fullPathH53 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-003/wehr3133_am_tuning_curve_20241212c.h5';
fullPathH54 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-004/wehr3133_am_tuning_curve_20241212d.h5';
fullPathH55 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-005/wehr3133_am_tuning_curve_20241212e.h5';
fullPathH56 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-006/wehr3133_am_tuning_curve_20241212f.h5';
fullPathH57 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-007/wehr3133_am_tuning_curve_20241212g.h5';
fullPathH58 = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-008/wehr3133_am_tuning_curve_20241212h.h5';

% tones = h5read(fullPathH5, '/resultsData/currentFreq');
% intensities = h5read(fullPathH5, '/resultsData/currentIntensity');
% times = h5read(fullPathH5, '/events/eventTime');
% codes = h5read(fullPathH5, '/events/eventCode');

% tones1 = h5read(fullPathH51, '/resultsData/currentFreq');
% intensities1 = h5read(fullPathH51, '/resultsData/currentIntensity');
% times1 = h5read(fullPathH51, '/events/eventTime');
% codes1 = h5read(fullPathH51, '/events/eventCode');

times1 = h5read(fullPathH51, '/events/eventTime');
times2 = h5read(fullPathH52, '/events/eventTime');
times3 = h5read(fullPathH53, '/events/eventTime');
times4 = h5read(fullPathH54, '/events/eventTime');
times5 = h5read(fullPathH55, '/events/eventTime');
times6 = h5read(fullPathH56, '/events/eventTime');
times7 = h5read(fullPathH57, '/events/eventTime');
times8 = h5read(fullPathH58, '/events/eventTime');

tones1 = h5read(fullPathH51, '/resultsData/currentFreq');
tones2 = h5read(fullPathH52, '/resultsData/currentFreq');
tones3 = h5read(fullPathH53, '/resultsData/currentFreq');
tones4 = h5read(fullPathH54, '/resultsData/currentFreq');
tones5 = h5read(fullPathH55, '/resultsData/currentFreq');
tones6 = h5read(fullPathH56, '/resultsData/currentFreq');
tones7 = h5read(fullPathH57, '/resultsData/currentFreq');
tones8 = h5read(fullPathH58, '/resultsData/currentFreq');