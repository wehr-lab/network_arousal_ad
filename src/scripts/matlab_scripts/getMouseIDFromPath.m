function mouseID = getMouseIDFromPath(datadir)

    mouseID = regexp(datadir, 'wehr\d+', 'match', 'once');

    if isempty(mouseID)
        error('Could not find mouse ID in path:\n%s', datadir);
    end

end
