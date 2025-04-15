function [] = Plot2P(varargin)

    % Plot two-photon tuning curve data and print to a pdf
    % You can either pass the full path to your 2P datadir (on the NAS), or you can run as a script by entering your plot directory below
    % Enter a directory and a cell number to plot (and NOT print) just that cell (NOT FINISHED YET SFM 12/17/24)
    
    if ~isempty(varargin)
        datadir = convertCharsToStrings(varargin{1});
    else
        datadir = pwd;
%         datadir = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr3133/12-12-24-002'; % enter directory to plot (if none explicitly passed)
    end
    
    if length(varargin) > 1
        CellToPlot = varargin{2};
    end

    switch getenv("USER")
        case 'wehr'
            figdir = fullfile(datadir, 'Figs'); % where would you like to save these tuning curves?
        otherwise
            figdir = '/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Figs'; % where would you like to save these tuning curves?
    end
    filepathparts = strsplit(datadir, '/'); mouseID = filepathparts{6}; sessionID = filepathparts{end};
    savename = fullfile(figdir, strcat(filepathparts{end-1}, '-', filepathparts{end}, '-TC-Mk2.ps'));

    behaviorMAT = dir(fullfile(datadir, 'wehr*.mat'));
    load(fullfile(datadir, behaviorMAT.name))
    FallPath = fullfile(datadir, '/suite2p/plane0/Fall.mat');
    load(FallPath)
    iscellList = load(FallPath, 'iscell');
    iscellList = iscellList.iscell;
    clear iscell

    behaviorH5 = dir(fullfile(datadir, 'wehr*.h5'));
    if isempty(behaviorH5)
        if exist(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID), 'dir')
            dateparts = strsplit(filepathparts{end}, '-'); month = dateparts{1}; day = dateparts{2}; year = strcat('20', dateparts{end});
            behaviorfiles = dir(fullfile('/Volumes/Projects/2P5XFAD/JarascopeData/behavior/', mouseID, strcat(mouseID, '_am_tuning_curve_', year, month, day, '_', sessionID)));
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
    spikesToPlot = spks(iscellLog, :);

    corrScalar = 0.7;
    cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);
%     cellsMean = mean(cellsToPlot, 2);

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
                    currTrace = (cellsToPlotCorr(currCell, currRange) - mean(cellsToPlotCorr(currCell, :)))/mean(cellsToPlotCorr(currCell, :));
                    meanRange(iTrial, :) = currTrace;
                end
                meanRanges{iTone, iInt} = meanRange';
            end
        end

        subplot1(2,6, 'Min', [0.05, 0.05], 'Gap', [0.01, 0.01]);
        fig = gcf; orient(fig, 'landscape');
        axes(fig, 'Position', [0.05, 0.05, 0.9, 0.9])
        title(sprintf('ROI %s Tuning Curve', num2str(currCell)), 'Position', [0.5, 1.02]); 
        text(0.5, -0.04, 'Time (in samples, 15.49 Hz)', 'HorizontalAlignment', 'center');
        axis off
        gcf;

        which_fig = 0;
        for iInt = 1:length(allInts)
            for iTone = 1:length(timestamps)
                which_fig = which_fig + 1;
                if sum(isnan(meanRanges{iTone, iInt}), 'all') ~= 0
                    meanRanges{iTone, iInt} = rmmissing(meanRanges{iTone, iInt}, 2);
                end
                meanTrace = mean(meanRanges{iTone, iInt}, 2);

                subplot1(which_fig); plot(meanRanges{iTone, iInt}, 'r', 'LineWidth', 1); plot(meanTrace, 'k', 'LineWidth', 2); xlim([1, 31]); xline(11, 'LineWidth', 1.5); ylim([-1.5, 10]); 
                if which_fig == 1
                    ylabel('dF/F - 70 dbSPL');
                elseif which_fig == 7
                    ylabel('dF/F - 50 dbSPL');
                end
                if which_fig == 1
                    xlabel('White Noise', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                elseif which_fig == 2
                    xlabel('2000 Hz', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                elseif which_fig == 3
                    xlabel('4000 Hz', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                elseif which_fig == 4
                    xlabel('8000 Hz', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                elseif which_fig == 5
                    xlabel('16000 Hz', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                elseif which_fig == 6
                    xlabel('32000 Hz', 'Position', [11, 10.65], 'HorizontalAlignment', 'center');
                end
                clear meanTrace
            end    
        end
        if length(varargin) <= 1
            if currCell == 1
                exportgraphics(gcf, savename);
            else
                exportgraphics(gcf, savename, 'Append', true);
            end
%             print(savename, '-dpsc2', '-append', '-bestfit');
            sprintf('On Cell %d / %d \n', currCell, size(cellsToPlotCorr, 1))
        end
        if exist('CellToPlot', 'var')
            break
        end
        clear meanRange meanRanges normRange currTrace
        close all
        
        
    end
