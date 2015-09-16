function deepTSvisualize(H)
    % Only the first 64 neurons will be plotted
    ha = tightSubplot(8,8,[.01 .03],[.1 .01],[.01 .01]);
    for ii = 1:64
        axes(ha(ii)); 
        plot(H(ii, :), 'LineWidth', 2);
    end
    set(ha(1:64),'XTickLabel',''); 
    set(ha(1:64),'YTickLabel','');
end

