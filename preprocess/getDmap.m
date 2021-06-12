image_path='../ShanghaiTech/part_B/test_data/images';
position_head_path='../ShanghaiTech/part_B/test_data/ground-truth';
dmap_path='../ShanghaiTech/part_B/test_data/Dmap';
for i=1:316
    imgPath=fullfile(image_path,num2str(i,'IMG_%d.jpg'));
    posPath=fullfile(position_head_path,num2str(i,'GT_IMG_%d.mat'));
    dmapPath=fullfile(dmap_path,num2str(i,'DMAP_%d.mat'));
    dmap=density(imgPath, posPath);
    save(dmapPath,'dmap');
    imagesc(dmap);
    %set(gca,'XTick',[]);
    %set(gca,'YTick',[]);
    %set(gca,'Position',[0 0 1 1]);
    %saveas(gcf,['./DMAP_',num2str(i),'.jpg'])
end