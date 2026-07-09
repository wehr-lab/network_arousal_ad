function [PupilDiameter] = ProcessPupilSFM(CamStruct,DLCProbabilityThreshold)
% Enter a mouse's directory path for processing all pupil cameras for all sessions, or pass a single session's path for just that directory
DirPath = convertCharsToStrings(CamStruct);

if length(strsplit(DirPath, '/')) == 5
    DirPath = fullfile(DirPath, 'PupilData'); % Add on whatever the directory name is for the pupil data
elseif length(strsplit(DirPath, '/')) == 6
else
    error("Can't parse directory path!")
end
allCams = dir(fullfile(DirPath, '*.csv'));

for iSess = 1:length(allCams)
    pupilPath = fullfile(allCams(iSess).folder, allCams(iSess).name);
    tempstr = strsplit(allCams(iSess).name, '_');
    mouseID = tempstr{1}; tempstr2 = tempstr(2); sessionID = string(tempstr(3)); 
    tempstr2 = convertStringsToChars(string(tempstr2)); currYear = tempstr2(end-3:end); currDay = tempstr2(end-5:end-4); currMonth = tempstr2(1:end-6);
    if length(currMonth) == 1
        currMonth = append("0", currMonth);
    end
    sessionDate = strcat(currMonth, '-', currDay, '-', currYear(3:end), '-', sessionID);
    fullpathOutput = fullfile(CamStruct, sessionDate, sprintf('%s_%s_PupilDiameter.mat', mouseID, sessionDate));

    dlcOut = readmatrix(pupilPath);
    px = zeros(size(dlcOut, 1), 8); 
    py = px; probability = px;
    j = 2:3:23;
    for i = 1:size(px, 2)
        px(:, i) = dlcOut(:, j(i));
        py(:, i) = dlcOut(:, j(i)+1);
        probability(:, i) = dlcOut(:, j(i)+2);
    end

    MajorDiameter = nan(length(px), 1); 
    MinorDiameter = MajorDiameter; Eccentricity = MinorDiameter; Indicatrix = MajorDiameter;

    for i = 1:length(px) %For each frame
    goodness = find(gt(probability(i, :),DLCProbabilityThreshold)); %Find Number of pts above DLCProbabilityThreshold
    if ge(length(goodness), 5) %If >= 5 good points, try to fit ellipse
        try
            ellipse_t = fit_ellipse(px(i, goodness), py(i, goodness));
            MajorDiameter(i) = ellipse_t.long_axis;
            if gt(MajorDiameter(i), 125) %discard values that are inconcievably large
                MajorDiameter(i) = nan;
            else
                MinorDiameter(i) = ellipse_t.short_axis;
                [Eccentricity(i)] = CalculateEccentricity(ellipse_t.a, ellipse_t.b);
                Indicatrix(i) = ellipse_t.short_axis/ellipse_t.long_axis;
            end
        end

    end
    end

%Package the values back into the camerastructure for output:
PupilDiameter.PupilMajor = MajorDiameter;
PupilDiameter.PupilMinor = MinorDiameter;
PupilDiameter.Eccentricity = Eccentricity;
PupilDiameter.Indicatrix = Indicatrix;
save(fullpathOutput, 'PupilDiameter');
end
end

function [e] = CalculateEccentricity(a, b)
c = real(sqrt((a * a)-(b * b)));
e = c/a;
end