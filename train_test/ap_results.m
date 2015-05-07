function res=ap_results(conf,labels,curvesdir,splitname,isvisible)
    assert(length(conf)==length(labels));
    labels(labels~=1) = -1 ;
    [r, p, info] = vl_pr(labels, conf);
    res.ap = info.ap; % AP
    res.appa = info.ap_interp_11; % PASCAL VOC AP (11pts)
    
    if nargin >= 3
        if nargin < 5
            isvisible = 0 ;
        end
        if ~isvisible
            figa=figure('Visible','off','Position', [10,10,500,500]);
        else
            figure('Position', [10,10,500,500]);
        end

        
        plot(r, p, 'LineWidth', 3);
        axis equal;
        axis([0,1,0,1]);
        xlabel('recall');
        ylabel('precision');
        title(sprintf('%s AP=%5.3f', splitname, res.ap));
        grid;
        set(gcf, 'PaperPositionMode', 'auto');
        print(gcf, [curvesdir '/' splitname], '-djpeg','-r100')
        
        if ~isvisible
            close(figa) ;
        end
    end
    
    
end
 