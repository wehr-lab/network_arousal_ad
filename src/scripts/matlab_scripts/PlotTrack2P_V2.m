function [] = PlotTrack2P_V2(varargin)

% Pass matched_suite2p directory (/Volumes/Projects/2P5XFAD/JarascopeData/[MOUSEID]/track2p/[TRACK2P IDENTIFIER]/matched_suite2p) 
% to plot cells tracked across sessions. Pass just the directory name to plot every cell and print to a .ps, pass with a number to plot a specific cell (ROI)
% 3rd input variable is a switch to plot spikes, use 1 to plot spikes, 0 to plot fluorescence traces

% Pass track2p directory (/Volumes/Projects/2P5XFAD/JarascopeData/[MOUSEID]/track2p/) to plot manually tracked cells across tracked sessions (if available)

if ~isempty(varargin)
    datadir = convertCharsToStrings(varargin{1});
else
    error('Gotta pass a directory to plot!')
end

if length(varargin) == 2
    CellToPlot = varargin{2};
end

datapathparts = strsplit(datadir, '/');
if strcmp(datapathparts{end}, 'matched_suite2p') == 0
    datadir = fullfile(datadir, 'matched_suite2p');
end
datapathparts = strsplit(datadir, '/');
mouseID = datapathparts{6}; 
figdir = '/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Figs'; % where would you like to save these figures?
basedir = '/Volumes/Projects/2P5XFAD/JarascopeData/'; % full data directory path to build subsequent filepaths from
if length(datapathparts) == 10
    trackedSession = datapathparts{9};
elseif length(datapathparts) == 9
    trackedSession = datapathparts{8};
else
    error('Make sure you pass the right directory for plotting!')
end
savename = fullfile(figdir, sprintf('%s-TrackedCombined-%s.pdf', mouseID, trackedSession));
numPlots = 0;

dirFilter = strcat(datadir, '/*-*');
matched_sessions = dir(dirFilter); 
%matched_sessions = matched_sessions(end-1:end);
for iSession = 1:length(matched_sessions)
    Sessions{iSession} = matched_sessions(iSession).name;
end

datetimeSort = datetime(Sessions, 'Format', 'MM-dd-yy-SSS');
Sessions = sort(datetimeSort);
for i = 1:length(Sessions)
    newSess{i} = char(Sessions(i));
end
Sessions = newSess;

for iDir = 1:length(Sessions)
    curr_session = fullfile(basedir, mouseID);
    tempH5 = dir(fullfile(curr_session, Sessions{iDir}, '*.h5'));
    if isempty(tempH5)
        if exist(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID), 'dir')
            dateparts = strsplit(Sessions{iDir}, '-'); month = dateparts{1}; day = dateparts{2}; year = strcat('20', dateparts{3}); sessionID = dateparts{end};
            behaviorfiles = dir(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID, strcat(mouseID, '_tones_and_wn_', year, month, day, '-', sessionID, '.h5')));
            if isempty(behaviorfiles)
                behaviorfiles = dir(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID, strcat(mouseID, '_am_tuning_curve_', year, month, day)));
                if length(behaviorfiles) > 1
                    error('Multiple behavior files are associated with this mouse and date, check with Sam (or the experimenter) what behavior file is correct')
                elseif isempty(behaviorfiles)
                    error("Can't find behavior file for this day, confirm that you ran 'sh copy_wehr_data_to_nas.sh' in the terminal on the two-photon behavior (Linux) computer to sync to the NAS")
                else
                    tempH5 = fullfile(behaviorfiles.folder, behaviorfiles.name);
                end
            else
                tempH5 = fullfile(behaviorfiles.folder, behaviorfiles.name);
            end
        else
            error("Can't find ANY behavior files associated with this mouse, confirm that you ran 'sh copy_wehr_data_to_nas.sh' in the terminal on the two-photon behavior (Linux) computer to sync to the NAS")
        end
    else
        tempH5 = fullfile(tempH5.folder, tempH5.name);
    end
    H5Paths{iDir} = tempH5;
    tempMat = dir(fullfile(curr_session, Sessions{iDir}, '*.mat'));
    MatPaths{iDir} = fullfile(basedir, mouseID, Sessions{iDir}, tempMat(1).name);
    
    tempFall = dir(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/Fall.mat'));
    if isempty(tempFall)
        F = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/F.npy'));
        Fneu = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/Fneu.npy'));
        iscell = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/iscell.npy'));
        spks = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/spks.npy'));
%         stat = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/stat.npy'));
%         ops = readNPY(fullfile(datadir, Sessions{iDir}, '/suite2p/plane0/ops.npy'));
%         save(fullfile(tempFall, 'Fall.mat'), 'F', 'Fneu', 'iscell', 'ops', 'spks', 'stat');
%         clear F Fneu iscell ops spks stat
    end
%     FallPaths{iDir} = fullfile(tempFall.folder, 'Fall.mat');
    tones = h5read(H5Paths{iDir}, '/resultsData/currentFreq');
    intensities = h5read(H5Paths{iDir}, '/resultsData/currentIntensity');
    allTones{iDir} = unique(tones);
    allInts{iDir} = unique(intensities); allInts{iDir} = flip(allInts{iDir});
    
    load(MatPaths{iDir})
    frames = info.frame;
    if rem(length(info.frame), length(tones)) == 2
    elseif rem(length(info.frame), length(tones)) == (length(tones) - 1)
    elseif ~(length(frames)/2 == length(tones))
        frames = frames(1:(end-2));
    end
    frameIndex = 1:2:length(frames);
    frames = frames(frameIndex);
    
    if length(tones) > length(frames)
        tones = tones(1:length(frames));
        intensities = intensities(1:length(frames));
    end
    
    nCond = 0;
    for iFreq = 1:length(allTones{iDir})
        for iInt = 1:length(allInts{iDir})
            nCond = nCond + 1;
            tempTimestamps = frames(tones == allTones{iDir}(iFreq));
            tempTimestampsInt = frames(intensities == allInts{iDir}(iInt));
            timestamps{iFreq, iInt} = tempTimestamps(ismember(tempTimestamps, tempTimestampsInt));
            nReps(nCond) = length(timestamps{iFreq, iInt});
        end
    end
    minReps = min(nReps);
    allTimestamps{iDir} = timestamps;
    allMinReps(iDir) = minReps;
    clear timestamps 
    
    if exist('iscell') == 1
        iscellList = iscell;
        clear iscell
    end
    iscellLog = logical(iscellList(:, 1)); iscellThresh = iscellList(:, 2);
    % Uncomment and enter a threshold value (0-1) to use Suite2P's likelihood value to select good cells
    % S2Pthresh = 0.95; % Suite2P likelihood threshold to use
    % iscellLog = iscellThresh >= S2Pthresh;
    if exist("CellToPlot") == 1
        cellsToPlot = F(CellToPlot, :);
        neucellsToPlot = Fneu(CellToPlot, :);
        goodSpikes = spks(CellToPlot, :);
    else
        cellsToPlot = F(iscellLog, :);
        neucellsToPlot = Fneu(iscellLog, :);
        goodSpikes = spks(iscellLog, :);
    end

    corrScalar = 0.7;
    cellsToPlotCorr{iDir} = cellsToPlot - (neucellsToPlot * corrScalar);
end

for iSess = 1:length(Sessions)
    IntsToLabel = allInts{iSess};
    for iAmp = 1:length(IntsToLabel)
        ylabels{iSess,iAmp} = sprintf('dF/F - %d dbSPL', IntsToLabel(iAmp));
    end
end
for iSess = 1:length(Sessions)
    TonesToLabel = allTones{iSess};
    for iFreq = 1:length(TonesToLabel)
        if TonesToLabel(iFreq) == -1
            xlabels{iSess,iFreq} = 'WN';
        else
            xlabels{iSess,iFreq} = sprintf('%.1f', TonesToLabel(iFreq)/1000);
        end
    end
end
cmap = jet(size(cellsToPlotCorr, 2));
curr_cell_plot = [];
if exist("CellToPlot")
    currCell = CellToPlot;
end

for i = 1:size(cellsToPlotCorr{1}, 1)
    if ~exist("CellToPlot")
        currCell = i;
    end
    for iDir = 1:length(Sessions)
        timestamps = allTimestamps{iDir};
        for iTone = 1:size(timestamps, 1)
            for iInt = 1:size(timestamps, 2)
                
                currTimestamps = timestamps{iTone, iInt};
                if length(currTimestamps) > allMinReps(iDir) || iDir == 1
                    currTimestamps = currTimestamps(1:allMinReps(iDir));
                end
                for iTrial = 1:length(currTimestamps)
                    currRange = (currTimestamps(iTrial) - 10):(currTimestamps(iTrial) + 20);
                    normRange = (currTimestamps(iTrial) - 11):(currTimestamps(iTrial) -1);
                    if ~isempty(currRange(currRange <= 0))
                        currRangeLog = currRange < 1;
                        currRange(currRangeLog) = 1;
                    end
                    if ~isempty(currRange(currRange > frames(end)))
                        currRangeLog = currRange > size(cellsToPlotCorr{iDir}, 2);
                        currRange(currRangeLog) = size(cellsToPlotCorr{iDir}, 2);
                    end
                    if sum(normRange <= 0) == length(normRange) || sum(normRange <= 0) >= 1
                        normRange = (currTimestamps(iTrial) + 21):(currTimestamps(iTrial) + 30);
                    end
                    if ~exist('CellToPlot')
                        currTrace = (cellsToPlotCorr{iDir}(currCell, currRange) - mean(cellsToPlotCorr{iDir}(currCell, :)))/mean(cellsToPlotCorr{iDir}(currCell, :));
                    else
                        currTrace = (cellsToPlotCorr{iDir}(1, currRange) - mean(cellsToPlotCorr{iDir}(1, :)))/mean(cellsToPlotCorr{iDir}(1, :));
                    end
                    meanRange(iTrial, :) = currTrace;
                end
                meanRanges{iTone, iInt} = meanRange';
            end
        end
        curr_cell_plot{iDir} = meanRanges;
    end
    
    subplot1(length(allInts{iDir}),length(allTones{iDir}), 'Min', [0.05, 0.05], 'Gap', [0.01, 0.01]);
    fig = gcf; orient(fig, 'landscape');
    axes(fig, 'Position', [0.05, 0.05, 0.9, 0.9])
    title(sprintf('ROI %s Tuning Curve - %s %s', num2str(currCell), datapathparts{8}, num2str(mouseID)), 'Position', [0.5, 1.02]);
    text(0.5, -0.04, 'Time (in samples, 15.49 Hz)', 'HorizontalAlignment', 'center');
    axis off
    gcf;
    
    for iCell = 1:length(curr_cell_plot)
        meanRanges = curr_cell_plot{iCell};
        which_fig = 0;
        for iInt = 1:size(timestamps, 2)
            for iTone = 1:size(timestamps, 1)
                which_fig = which_fig + 1;
                if sum(isnan(meanRanges{iTone, iInt}), 'all') ~= 0
                    meanRanges{iTone, iInt} = rmmissing(meanRanges{iTone, iInt}, 2);
                end
                meanTrace = mean(meanRanges{iTone, iInt}, 2);
                
                subplot1(which_fig);
                hold on; plot(meanTrace, 'Color', cmap(iCell,:), 'LineWidth', 1.5);
                ylims = [-1.5, 10];
                ylim(ylims);
                xlim([1, 31]);
                xline(11, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
                if iTone == 1
                    ylabel(ylabels{iDir, iInt});
                end
                if iInt == 1
                    xLabYPos = ylims(2) + 1;
                    xlabel(xlabels{iDir, iTone}, 'Position', [11, xLabYPos], 'HorizontalAlignment', 'center');
                end
                clear meanTrace ydots
            end
        end
    end
    
    if ~exist(savename)
        exportgraphics(gcf, savename, 'ContentType', 'image', 'PreserveAspectRatio', 'on');
    else
        exportgraphics(gcf, savename, 'ContentType', 'image', 'PreserveAspectRatio', 'on', 'Append', true);
    end
    close all
    clear meanRanges meanRange meanTrace %curr_cell_plot timestamps currTimestamps currRange normRange currTrace
end