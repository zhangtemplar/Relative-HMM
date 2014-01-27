% get the accent data
url_root='http://accent.gmu.edu/browse_language.php?function=find&language=mandarin';
url_base='http://accent.gmu.edu/browse_language.php?function=detail&speakerid=';
url_link='http://chnm.gmu.edu/accent/soundtracks/mandarin';
str=urlread(url_root);
% search for the link
url_pattern='<a href="browse_language.php?function=detail&speakerid=';
link_pattern='http://chnm.gmu.edu/accent/soundtracks/mandarin';
index_start=strfind(str, url_pattern);
index_end=strfind(str, '">mandarin');
for k=1: length(index_start)
    speaker_id=str(index_start(k)+length(url_pattern): index_end(k)-1);
    url_speaker=[url_base speaker_id];
    str_speaker=urlread(url_speaker);
    % search for the audio file
    idx1=strfind(str_speaker, link_pattern);
    idx2=strfind(str_speaker, '.mov');
    urlwrite([url_link str_speaker(idx1+length(link_pattern): idx2) 'mov'], ['mandarin' speaker_id '.mov']);
end
