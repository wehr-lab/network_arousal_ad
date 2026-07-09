function fig = Plot2PCurves(plotDir, iCell)

plotDeconvolved = 0; %logical for plotting deconvolved signal

matH5 = dir(fullfile(plotDir,'wehr*.mat'));
load(matH5.name)
FallDir = dir(fullfile(plotDir,'Fall.mat'));
if isempty(FallDir)
    currDirContents = dir(fullfile(plotDir, 'suite2p'));
    if ~isempty(currDirContest)
        s2pDirContents = dir(fullfile(plotDir, '~/suite2p/plane0'));
        if ~isempty(s2pDirContents)
            plane0DirContents = dir(fullfile(plotDir, '~/suite2p/plane0/Fall.mat'));
            if ~isempty(plane0DirContents)
                load(fullfile(plotDir, 'Fall.mat'))
                iscellList = load('Fall.mat', 'iscell');
                iscellList = iscellList.iscell(:, 1);
                clear iscell
            else
                error('Could not locate F/all statistics in this directory!')
            end
        else
            error('Could not locate F/all statistics in this directory!')
        end
    else
        error('Could not locate F/all statistics in this directory!')
    end
else
    load(fullfile(plotDir,'Fall.mat'))
    iscellList = load(fullfile(plotDir,'Fall.mat'), 'iscell');
    iscellList = iscellList.iscell(:, 1);
    clear iscell
end

behaviorH5 = dir(fullfile(plotDir, 'wehr*.h5'));
if isempty(behaviorH5)
    currDirContents = dir(fullfile(plotDir,'suite2p'));
    if ~isempty(currDirContents)
        cd('suite2p')
        s2pDirContents = dir(fullfile(plotDir,'plane0'));
        if ~isempty(s2pDirContents)
            cd('plane0')
            plane0DirContents = dir(fullfile(plotDir,'wehr*.h5'));
            if ~isempty(plane0DirContents)
                fullpathH5 = fullfile(pwd, plane0DirContents.name);
            else
                error('Could not locate H5 Behavior file, make sure you transferred it over from the "behavior" repository!')
            end
        end
    end
else
    fullpathH5 = fullfile(currDir, behaviorH5.name);
end

tones = h5read(fullpathH5, '/resultsData/currentFreq');
intensities = h5read(fullPathH5, '/resultsData/currentIntensity');
allTones = unique(tones);
allInts = unique(intensities); allInts = flip(allInts);

frames = info.frame;
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

iscellLog = logical(iscellList(:, 1));
cellsToPlot = F(iscellLog, :);
neucellsToPlot = Fneu(iscellLog, :);
spikesToPlot = spks(iscellLog, :);

corrScalar = 0.7;
cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);
cellsMean = mean(cellsToPlot, 2);

for i = 1:length(timestamps)
    timestampCount(i) = length(timestamps{1, i});
end
minTimestamps = min(timestampCount);
cmap = jet(max(timestampCount));

currCell = iCell;
for iTone = 1:length(timestamps)
    currTimestamps = timestamps{iTone};
    if length(currTimestamps) ~= minTimestamps
        currTimestamps = currTimestamps(1:minTimestamps);
    end
    for iTrial = 1:length(currTimestamps)
        currRange = (currTimestamps(iTrial) - 10):(currTimestamps(iTrial) + 20);
        normRange = (currTimestamps(iTrial) - 11):(currTimestamps(iTrial) - 1);
        if ~isempty(currRange(currRange <= 0))
            currRangeLog = currRange < 1;
            currRange(currRangeLog) = 1;
        end
        if ~isempty(normRange(normRange <= 0))
            normRange = normRange(normRange > 0);
        end
        currTrace = (cellsToPlotCorr(currCell, currRange) - mean(cellsToPlotCorr(currCell, normRange)))/mean(cellsToPlotCorr(currCell, normRange));
        meanRange(iTrial, :) = currTrace;
        if plotDeconvolved == 1
            if iTrial == 1
                figDecon = figure;
            end
            plot(spikesToPlot(currCell, currRange), 'LineWidth', 2, 'Color', cmap(iTrial,:));
            hold on
        end
    end
    if plotDeconvolved == 1
        xlim([1, 31])
        xlabel('Time (in samples, 15.49 Hz)')
        ylabel('Deconvolved Spike Signal')
        xline(11, 'LineWidth', 1.5)
        ylim([0, inf])
        title(sprintf('ROI %s Deconvolved Spikes - %s Hz', num2str(currCell), num2str(allTones(iTone))));
        hold off
    end
    
    fig = figure;
    meanTrace = mean(meanRange, 1);
    hold on
    plot(meanRange', 'k', 'LineWidth', 1)
    plot(meanTrace, 'r', 'LineWidth', 2)
    xlim([1, 31])
    xlabel('Time (in samples, 15.49 Hz)')
    ylabel('Mean dF/F')
    xline(11, 'LineWidth', 1.5)
    ylim([-1.5, 10])
    title(sprintf('ROI %s Mean Trace - %s Hz', num2str(currCell), num2str(allTones(iTone))));
    clear meanRange
end