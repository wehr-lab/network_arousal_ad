% 
% mousedir = '/Volumes/Projects/2P5XFAD/JarascopeData/wehr5142'; %path to mouse data directory
% mouseNUM = 1;
% tempstr = strsplit(mousedir, 'wehr');
% mouseID = str2num(tempstr{end});
% 
% allsessions = dir(fullfile(mousedir, '*-000'));
% allsessions = allsessions(2:end);
% allsessions = [allsessions(1:2); allsessions(4:end)];
% 
% for i = 1:length(allsessions)
%     currFall = fullfile(allsessions(i).folder, allsessions(i).name, '/suite2p/plane0/iscell.npy');
%     iscellSTAT = readNPY(currFall);
%     manualcellcount = iscellSTAT(:,1);
%     threshcellcount = iscellSTAT(:,2);
%     totalnumcells{i} = length(manualcellcount(manualcellcount == 1));
%     threshnumcells{i} = length(threshcellcount(threshcellcount >= 0.9));
% end
% 
% TotalNumCellsByMouse{mouseNUM} = totalnumcells;
% ThreshNumCellsByMouse{mouseNUM} = threshnumcells;
% mouselist{mouseNUM} = mouseID;
% clear totalnumcells threshnumcells
posCond = 0;
negCond = 0;

load('/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/CellsBySession.mat')
for iMouse = 1:length(mouselist)
    currCounts = cell2mat(TotalNumCellsByMouse{iMouse});
    currAges = MouseSessionAges{iMouse};
    currMouse = mouselist{iMouse};
    
    if iMouse == 1
        figure
    else
        hold on
    end
    if condition{iMouse} == '+'
        posCond = posCond + 1;
        plot(currAges, currCounts , 'LineWidth', 2, 'Color', 'red');
        if posCond == 1
            legendMap(1) = plot(currAges, currCounts , 'LineWidth', 2, 'Color', 'red');
        end
    else
        negCond = negCond + 1;
        plot(currAges, currCounts , 'LineWidth', 2, 'Color', 'black');
        if negCond == 1
            legendMap(2) = plot(currAges, currCounts , 'LineWidth', 2, 'Color', 'black');
        end
    end
end
xlim([50 250])
ylim([0 1600])
legendlabels = {'5XFAD', 'Control'};
legend(legendMap, legendlabels, 'Location', 'Northeast')
xlabel('Mouse Age (Days)')
ylabel('Cell Count')
title('Cell Counts by Age in 2P')
    