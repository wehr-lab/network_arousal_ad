function [] = Plot2P(varargin)

    % Plot two-photon tuning curve data and print to a pdf
    % You can either pass the full path to your 2P datadir (on the NAS), or you can run as a script by entering your plot directory below
    % Enter a directory and a cell number to plot (and NOT print) just that cell
    % Enter a third variable as a logical to plot spikes. 1 to plot spikes, 0 to plot fluorescence traces
    
    if ~isempty(varargin)
        datadir = convertCharsToStrings(varargin{1});
    else
        error('Gotta pass a directory to plot!') % enter directory to plot (if none explicitly passed)
    end
    
    if length(varargin) == 2
        CellToPlot = varargin{2};
    end
    
    if length(varargin) == 3
        if ~isempty(varargin{2})
            CellToPlot = varargin{2};
        end
        spikeLog = varargin{3};
    else
        spikeLog = 0;                                                       % Logical variable to switch between plotting luminence traces or deconvolved spikes
    end
    smin = 0.5;                                                             % Threshold for spikes (smin * maxvalue per cell = minimum spike threshold)

    figdir = '/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Figs'; % where would you like to save these tuning curves?
    filepathparts = strsplit(datadir, '/'); mouseID = filepathparts{6}; sessionID = filepathparts{end};
    savename = fullfile(figdir, strcat(filepathparts{end-1}, '-', filepathparts{end}, '-TC.pdf'));
    numPlots = 0;

    behaviorMAT = dir(fullfile(datadir, 'wehr*.mat'));
    load(fullfile(datadir, behaviorMAT(1).name))
    FallPath = fullfile(datadir, '/suite2p/plane0/Fall.mat');
    load(FallPath)
    iscellList = load(FallPath, 'iscell');
    iscellList = iscellList.iscell;
    clear iscell

    behaviorH5 = dir(fullfile(datadir, 'wehr*.h5'));
    if isempty(behaviorH5)
        if exist(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID), 'dir')
            dateparts = strsplit(filepathparts{end}, '-'); month = dateparts{1}; day = dateparts{2}; year = strcat('20', dateparts{3}); sessionID2 = dateparts{end};
            behaviorfiles = dir(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID, strcat(mouseID, '_tones_and_wn_', year, month, day, '-', sessionID2, '.h5')));
            if isempty(behaviorfiles)
                behaviorfiles = dir(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID, strcat(mouseID, '_am_tuning_curve_', year, month, day)));
                if length(behaviorfiles) > 1
                    error('Multiple behavior files are associated with this mouse and date, check with Sam (or the experimenter) what behavior file is correct')
                elseif isempty(behaviorfiles)
                    error("Can't find behavior file for this day, confirm that you ran 'sh copy_wehr_data_to_nas.sh' in the terminal on the two-photon behavior (Linux) computer to sync to the NAS")
                else
                    fullPathH5 = fullfile(behaviorfiles.folder, behaviorfiles.name);
                end
            else
                fullPathH5 = fullfile(behaviorfiles.folder, behaviorfiles.name);
            end
        else
            error("Can't find ANY behavior files associated with this mouse, confirm that you ran 'sh copy_wehr_data_to_nas.sh' in the terminal on the two-photon behavior (Linux) computer to sync to the NAS")
        end
    else
        fullPathH5 = fullfile(datadir, behaviorH5.name);
    end
    
    tones = h5read(fullPathH5, '/resultsData/currentFreq');
    intensities = h5read(fullPathH5, '/resultsData/currentIntensity');
    allTones = unique(tones);
    allInts = unique(intensities); allInts = flip(allInts);
%     tones = tones((end-136):end);
%     intensities = intensities((end-136):end); % +137; ignore this, manual coding to overcome TasKontrol appending data sessions

    frames = info.frame; %tones = tones(1:end-1); intensities = intensities(1:end-1);
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
    for iFreq = 1:length(allTones)
        for iInt = 1:length(allInts)
            nCond = nCond + 1;
            tempTimestamps = frames(tones == allTones(iFreq));
            tempTimestampsInt = frames(intensities == allInts(iInt));
            timestamps{iFreq, iInt} = tempTimestamps(ismember(tempTimestamps, tempTimestampsInt));
            nReps(nCond) = length(timestamps{iFreq, iInt});
        end
    end
    minReps = min(nReps);

    iscellLog = logical(iscellList(:, 1)); iscellThresh = iscellList(:, 2);
    % Uncomment and enter a threshold value (0-1) to use Suite2P's likelihood value to select good cells
    % S2Pthresh = 0.95; % Suite2P likelihood threshold to use
    % iscellLog = iscellThresh >= S2Pthresh; 
    cellsToPlot = F(iscellLog, :);
    neucellsToPlot = Fneu(iscellLog, :);
    
    if spikeLog == 1
        spikesToPlot = spks(iscellLog, :);
        for iCell = 1:size(spikesToPlot,1)
            maxResp = max(spikesToPlot(iCell,:));
            rast = spikesToPlot(iCell,:) >= (maxResp * smin);
            totalSpikeCounts(iCell) = sum(rast);
            goodSpikes(iCell, :) = rast;
        end
    end

    corrScalar = 0.7;
    cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);
%     cellsMean = mean(cellsToPlot, 2);

    IntsToLabel = allInts;
    for iAmp = 1:length(IntsToLabel)
        ylabels{iAmp} = sprintf('dF/F - %d dbSPL', IntsToLabel(iAmp));
    end
    TonesToLabel = allTones;
    for iFreq = 1:length(TonesToLabel)
        if TonesToLabel(iFreq) == -1
            xlabels{iFreq} = 'WN';
        else
            xlabels{iFreq} = sprintf('%.1f', TonesToLabel(iFreq)/1000);
        end
    end

    for currCell = 1:size(cellsToPlotCorr, 1)
        if length(varargin) == 2
            currCell = CellToPlot;
        end
        for iTone = 1:length(timestamps)
            for iInt = 1:length(allInts)
                currTimestamps = timestamps{iTone, iInt};
                if length(currTimestamps) > minReps
                    currTimestamps = currTimestamps(1:minReps);
                end
                for iTrial = 1:length(currTimestamps)
                    currRange = (currTimestamps(iTrial) - 10):(currTimestamps(iTrial) + 20);
                    normRange = (currTimestamps(iTrial) - 11):(currTimestamps(iTrial) -1);
                    if ~isempty(currRange(currRange <= 0))
                        currRangeLog = currRange < 1;
                        currRange(currRangeLog) = 1;
                    end
                    if ~isempty(currRange(currRange > frames(end)))
                        currRangeLog = currRange > size(cellsToPlotCorr, 2);
                        currRange(currRangeLog) = size(cellsToPlotCorr, 2);
                    end
                    if sum(normRange <= 0) == length(normRange) || sum(normRange <= 0) >= 1
                        normRange = (currTimestamps(iTrial) + 21):(currTimestamps(iTrial) + 30);
                    end
%                     currTrace = (cellsToPlotCorr(currCell, currRange) - mean(cellsToPlotCorr(currCell, normRange)))/mean(cellsToPlotCorr(currCell, normRange));
                    if spikeLog == 1
                        currSpikes = goodSpikes(currCell, currRange);
                        tempSpikes(iTrial,:) = currSpikes;
                    else
                        currTrace = (cellsToPlotCorr(currCell, currRange) - mean(cellsToPlotCorr(currCell, :)))/mean(cellsToPlotCorr(currCell, :));
                        meanRange(iTrial, :) = currTrace;
                    end
                end
                if spikeLog == 1
                    spikeResp{iTone, iInt} = tempSpikes;
                else
                    meanRanges{iTone, iInt} = meanRange';
                end
            end
        end

        subplot1(length(allInts), length(allTones), 'Min', [0.05, 0.05], 'Gap', [0.01, 0.01]);
        fig = gcf; orient(fig, 'landscape');
        axes(fig, 'Position', [0.05, 0.05, 0.9, 0.9])
        if spikeLog ~= 1
            title(sprintf('ROI %s Tuning Curve - Sess. %s', num2str(currCell), sessionID), 'Position', [0.5, 1.02]);
        else
            title(sprintf('ROI %s PSTH - Sess. %s', num2str(currCell), sessionID), 'Position', [0.5, 1.02]);
        end
        text(0.5, -0.04, 'Time (in samples, 15.49 Hz)', 'HorizontalAlignment', 'center');
        axis off
        gcf;

        which_fig = 0;
        for iInt = 1:length(allInts)
            for iTone = 1:length(allTones)
                which_fig = which_fig + 1;
                if spikeLog ~= 1
                    if sum(isnan(meanRanges{iTone, iInt}), 'all') ~= 0
                        meanRanges{iTone, iInt} = rmmissing(meanRanges{iTone, iInt}, 2);
                    end
                    meanTrace = mean(meanRanges{iTone, iInt}, 2);
                end

                subplot1(which_fig); %plot(meanRanges{iTone, iInt}, 'r', 'LineWidth', 1); 
                if spikeLog == 1
                    hold on; histogram('BinEdges', [0.5:31.5], 'BinCounts',sum(spikeResp{iTone,iInt},1));
                    histo = findobj(gca);
                    histo(2).FaceColor = [0 0 0];
                    ylims = [0, 10];
                    ylim(ylims);
                    xlim([1, 31]);
                    [~, xdots] = find(sum(spikeResp{iTone, iInt},1));
                    if ~isempty(xdots)
                        ydots = (0.67 * ylims(2)) + random('normal', 0, 1, [1 length(xdots)]);
                        hold on; scatter(xdots, ydots, 30, '.k');
                    end
                    xlim([1, 31]); xline(11, 'LineWidth', 1.5);
                else
                    hold on; plot(meanTrace, 'k', 'LineWidth', 2); 
                    xlim([1, 31]); 
                    xline(11, 'LineWidth', 1.5); 
                    ylim([-1.5, 10]); 
                end
                if iTone == 1
                    ylabel(ylabels{iInt});
                end
                if iInt == 1
                    xlabel(xlabels{iTone}, 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                end
                clear meanTrace
            end    
        end
        if length(varargin) <= 1
            if numPlots == 0
                exportgraphics(gcf, savename,'ContentType', 'image', 'BackgroundColor', 'white', 'PreserveAspectRatio', 'on');
                numPlots = numPlots + 1;
            else
                exportgraphics(gcf, savename, 'ContentType', 'image', 'BackgroundColor', 'white', 'PreserveAspectRatio', 'on', 'Append', true);
            end
            % print(savename, '-dpsc2', '-append', '-bestfit');
            sprintf('On Cell %d / %d \n', currCell, size(cellsToPlotCorr, 1))
        end
        if exist('CellToPlot', 'var')
            break
        end
        clear meanRange meanRanges normRange currTrace
        close all
        
        
    end
