function [] = calculateDriftBestInt(varargin)

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
sessLog = strfind(datapathparts, 'track2p-w');
temp = [];
for i = 1:length(sessLog)
    if isempty(sessLog{i})
        temp(i) = 0;
    else
        temp(i) = sessLog{i};
    end
end
sessLog = logical(temp);
trackSessionID = string(datapathparts(sessLog));
mouseID = datapathparts{6}; 
figdir = '/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Figs'; % where would you like to save these figures?
basedir = '/Volumes/Projects/2P5XFAD/JarascopeData/'; % full data directory path to build subsequent filepaths from
if length(datapathparts) == 10
    trackedSession = datapathparts{9};
elseif length(datapathparts) == 9
    trackedSession = datapathparts{8};
else
    % error('Make sure you pass the right directory for plotting!')
end
savename = fullfile(figdir, sprintf('%s-DriftPlots-%s.pdf', mouseID, trackSessionID));

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
    cellsToPlot = F(iscellLog, :);
    neucellsToPlot = Fneu(iscellLog, :);
    goodSpikes = spks(iscellLog, :);

    corrScalar = 0.7;
    cellsToPlotCorr{iDir} = cellsToPlot - (neucellsToPlot * corrScalar);
end

for iSess = 1:length(Sessions)
    IntsToLabel = allInts{iSess};
    for iAmp = 1:length(IntsToLabel)
        ylabels{iSess,iAmp} = sprintf('dF/F - %d dbSPL', IntsToLabel(iAmp));
    end
end
% for iSess = 1:length(Sessions)
    TonesToLabel = allTones{iSess};
    for iFreq = 1:length(TonesToLabel)
        if TonesToLabel(iFreq) == -1
            % xlabels{iSess,iFreq} = 'WN';
            xlabels{iFreq} = 'WN';
        else
            % xlabels{iSess,iFreq} = sprintf('%.1f', TonesToLabel(iFreq)/1000);
            xlabels{iFreq} = sprintf('%.1f', TonesToLabel(iFreq)/1000);
        end
    end
% end

for currCell = 1:size(cellsToPlotCorr{1}, 1)
    if length(varargin) == 2
        currCell = CellToPlot;
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
                    %currTrace = (cellsToPlotCorr{iDir}(currCell, currRange) - mean(cellsToPlotCorr{iDir}(currCell, normRange)))/mean(cellsToPlotCorr{iDir}(currCell, normRange));
                    % should the norm range be the entire session (TC + Resting State) or just the stimulus presentation?
                    currTrace = (cellsToPlotCorr{iDir}(currCell, currRange) - mean(cellsToPlotCorr{iDir}(currCell, :)))/mean(cellsToPlotCorr{iDir}(currCell, :));
                    contTrace = (cellsToPlotCorr{iDir}(currCell, (currRange+27882)) - mean(cellsToPlotCorr{iDir}(currCell, :)))/mean(cellsToPlotCorr{iDir}(currCell, :));
                    meanRange(iTrial, :) = currTrace;
                    meanContRange(iTrial, :) = contTrace;
                end
                meanRanges{iTone, iInt} = meanRange';
                meanContRanges{iTone, iInt} = meanContRange';
            end
        end
        allCorrMax(iDir) = max((cellsToPlotCorr{iDir}(currCell, :) - mean(cellsToPlotCorr{iDir}(currCell, :)))/mean(cellsToPlotCorr{iDir}(currCell, :)));
        allCorrMin(iDir) = min((cellsToPlotCorr{iDir}(currCell, :) - mean(cellsToPlotCorr{iDir}(currCell, :)))/mean(cellsToPlotCorr{iDir}(currCell, :)));
        allSilentAvg(iDir) = mean((cellsToPlotCorr{iDir}(currCell, 27883:end) - mean(cellsToPlotCorr{iDir}(currCell, 27883:end)))/mean(cellsToPlotCorr{iDir}(currCell, 27883:end)));

        for iTone = 1:size(meanRanges, 1)
            for iInt = 1:size(meanRanges, 2)
                maxResponse(iTone, iInt) = max(mean(meanRanges{iTone, iInt}, 2));
            end
        end
        allMaxResponses{iDir} = maxResponse;

    end
    %significance tests comparing mean peak responses to silent avg, what test to use?
    for iSes = 1:length(allMaxResponses)
        currSess = allMaxResponses{iSes};
        [tempMax, tempMaxInt] = max(currSess, [], 'all');
        [bestFreq, bestInt] = find((currSess == tempMax));
        all_CurrMaxRespInt(iSes) = bestInt;
        all_CurrMaxResp(iSes) = tempMax;
    end

    curr_maxRespInt = mode(all_CurrMaxRespInt);
    if length(curr_maxRespInt) > 1
        [~, peakResp] = max(all_CurrMaxResp);
        curr_maxRespInt = all_CurrMaxRespInt(peakResp);
    end
    
    cmap = jet(length(Sessions));
    figure
    for i = 1:length(Sessions)
        plot(allMaxResponses{i}(:, curr_maxRespInt), '-', 'Color', cmap(i,:), 'LineWidth', 1.5);
        hold on
    end
    hold off
    xlim([1, 18]);
    ylim([-1, 15]);
    xticks(1:19)
    xticklabels(xlabels);
    xlabel('Tones')
    ylabel('Peak Mean Response \DeltaF/F')
    title(sprintf('%s - %s - ROI %d Tuning Drift', mouseID, trackSessionID, currCell))
    for numSess = 1:length(Sessions)
        colorbar_labels{numSess} = sprintf('S%d', numSess);
    end
    labels = sprintf('%d dbSPL', allInts{1}(curr_maxRespInt));
    colormap jet
    label_interval = 1/length(Sessions);
    colorbar('Ticks', 0:label_interval:(1-label_interval), 'TickLabels', colorbar_labels);
    legend(labels, 'Location', 'northwest');

    if exist('CellToPlot')
        break
    end
    if ~exist(savename)
        exportgraphics(gcf, savename, 'ContentType', 'image', 'PreserveAspectRatio', 'on');
    else
        exportgraphics(gcf, savename, 'ContentType', 'image', 'PreserveAspectRatio', 'on', 'Append', true);
    end
    close all
    clear meanRange meanRanges allMaxResponses normRange currTrace timestamps currTimestamps currRange 
end