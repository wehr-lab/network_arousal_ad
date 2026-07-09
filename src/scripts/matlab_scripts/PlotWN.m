function fig = PlotWN(plotDir, iCell)

% plotDeconvolved = 0; %logical for plotting deconvolved signal
currDir = plotDir; % enter directory containing .sbx and .mat recording files

cd(currDir)

matH5 = dir('wehr*.mat');
load(matH5.name)
FallDir = dir('Fall.mat');
if isempty(FallDir)
    currDirContents = dir('suite2p');
    if ~isempty(currDirContest)
        s2pDirContents = dir('~/suite2p/plane0');
        if ~isempty(s2pDirContents)
            plane0DirContents = dir('~/suite2p/plane0/Fall.mat');
            if ~isempty(plane0DirContents)
                load('Fall.mat')
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
    load('Fall.mat')
    iscellList = load('Fall.mat', 'iscell');
    iscellList = iscellList.iscell(:, 1);
    clear iscell
end

behaviorH5 = dir(fullfile(plotDir, 'wehr*.h5'));
if isempty(behaviorH5)
    currDirContents = dir('suite2p');
    if ~isempty(currDirContents)
        cd('suite2p')
        s2pDirContents = dir('plane0');
        if ~isempty(s2pDirContents)
            cd('plane0')
            plane0DirContents = dir('wehr*.h5');
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
allTones = unique(tones);

frames = info.frame;
if ~(length(frames)/2 == length(tones))
    frames = frames(1:(end-2));
end
frameIndex = 1:2:length(frames);
frames = frames(frameIndex);

iscellLog = logical(iscellList(:, 1));
cellsToPlot = F(iscellLog, :);
neucellsToPlot = Fneu(iscellLog, :);
spikesToPlot = spks(iscellLog, :);

corrScalar = 0.7;
cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);
cellsMean = mean(cellsToPlot, 2);

for iNeuron = 1:sum(iscellLog)
    dFFcells(iNeuron, :) = (cellsToPlotCorr(iNeuron, :) - cellsMean(iNeuron))/cellsMean(iNeuron);
end

cmap = jet(length(frames));

currCell = iCell;
for iTone = 1:length(frames)
    currTimestamps = frames(iTone);
    currRange = (currTimestamps - 10):(currTimestamps + 20);
        if ~isempty(currRange(currRange <= 0))
            currRangeLog = currRange < 1;
            currRange(currRangeLog) = 1;
        end
        if plotDeconvolved == 1
            if iTrial == 1
                figDecon = figure;
            end
            plot(spikesToPlot(currCell, currRange), 'LineWidth', 2, 'Color', cmap(iTrial,:));
            hold on
        end
    meanRange(iTone, :) = dFFcells(currCell, currRange);
%     if plotDeconvolved == 1
%         xlim([1, 31])
%         xlabel('Time (in samples, 15.49 Hz)')
%         ylabel('Deconvolved Spike Signal')
%         xline(11, 'LineWidth', 1.5)
%         ylim([0, inf])
%         title(sprintf('ROI %s Deconvolved Spikes - %s Hz', num2str(currCell), num2str(allTones(iTone))));
%         hold off
%     end
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
title(sprintf('ROI %s White Noise Response - %s Hz', num2str(currCell), num2str(allTones(iTone))));