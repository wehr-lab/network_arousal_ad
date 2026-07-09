function sessionID = getSessionIDFromPath(datadir)
%GETSESSIONIDFROMPATH Extract session ID from a Jarascope path.
%
% Example:
%
%   getSessionIDFromPath( ...
%       '/Volumes/Projects/2P5XFAD/JarascopeData/wehr6026/05-18-26-000')
%
%   returns:
%       '05-18-26-000'
%
% Example:
%
%   getSessionIDFromPath( ...
%       '/Volumes/Projects/2P5XFAD/JarascopeData/wehr6026/' ...
%       'track2p/track2p-w12/matched_suite2p/05-20-26-000')
%
%   returns:
%       '05-20-26-000'

    sessionID = regexp(datadir, ...
        '\d{2}-\d{2}-\d{2}-\d{3}', ...
        'match', ...
        'once');

    if isempty(sessionID)
        error('Could not find session ID in path:\n%s', datadir);
    end

end