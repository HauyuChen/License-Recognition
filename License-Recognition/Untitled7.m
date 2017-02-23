clear ;
close all;
%web -browser http://www.ilovematlab.cn/thread-23229-1-1.html
%Step1 ��ȡͼ��   װ��������ɫͼ����ʾԭʼͼ��
Scolor = imread('3.jpg');%imread������ȡͼ���ļ�
%����ɫͼ��ת��Ϊ�ڰײ���ʾ
Sgray = rgb2gray(Scolor);%rgb2grayת���ɻҶ�ͼ
figure,imshow(Scolor),title('ԭʼ��ɫͼ��');%figure����ͬʱ��ʾ����ͼ��
figure,imshow(Sgray),title('ԭʼ�ڰ�ͼ��');
%Step2 ͼ��Ԥ����   ��Sgray ԭʼ�ڰ�ͼ����п������õ�ͼ�񱳾�
s=strel('disk',13);%strei����
Bgray=imopen(Sgray,s);%��sgray sͼ��
figure,imshow(Bgray);title('����ͼ��');%�������ͼ��
%��ԭʼͼ���뱳��ͼ������������ǿͼ��
Egray=imsubtract(Sgray,Bgray);%����ͼ���
figure,imshow(Egray);title('��ǿ�ڰ�ͼ��');%����ڰ�ͼ��
%Step3 ȡ�������ֵ����ͼ���ֵ��
fmax1=double(max(max(Egray)));%egray�����ֵ�����˫������
fmin1=double(min(min(Egray)));%egray����Сֵ�����˫������
level=(fmax1-(fmax1-fmin1)/3)/255;%��������ֵ
bw22=im2bw(Egray,level);%ת��ͼ��Ϊ������ͼ��
bw2=double(bw22);
%Step4 �Եõ���ֵͼ�������ղ��������˲�
figure,imshow(bw2);title('ͼ���ֵ��');%�õ���ֵͼ��
grd=edge(bw2,'canny')%��canny����ʶ��ǿ��ͼ���еı߽�
figure,imshow(grd);title('ͼ���Ե��ȡ');%���ͼ���Ե
bg1=imclose(grd,strel('rectangle',[5,19]));%ȡ���ο�ı�����
figure,imshow(bg1);title('ͼ�������[5,19]');%����������ͼ��
bg3=imopen(bg1,strel('rectangle',[5,19]));%ȡ���ο�Ŀ�����
figure,imshow(bg3);title('ͼ������[5,19]');%����������ͼ��
bg2=imopen(bg3,strel('rectangle',[19,1]));%ȡ���ο�Ŀ�����
figure,imshow(bg2);title('ͼ������[19,1]');%����������ͼ��
%Step5 �Զ�ֵͼ�����������ȡ���������������������������������������Ƚϣ���ȡ��������
[L,num] = bwlabel(bg2,8);%��ע������ͼ���������ӵĲ���
Feastats = regionprops(L,'basic');%����ͼ������������ߴ�
Area=[Feastats.Area];%�������
BoundingBox=[Feastats.BoundingBox];%[x y width height]���ƵĿ�ܴ�С
RGB = label2rgb(L, 'spring', 'k', 'shuffle'); %��־ͼ����RGBͼ��ת��
figure,imshow(RGB);title('ͼ���ɫ���');%�����ܵĲ�ɫͼ��
lx=0;
for l=1:num
    width=BoundingBox((l-1)*4+3);%��ܿ�ȵļ���
    hight=BoundingBox((l-1)*4+4);%��ܸ߶ȵļ���
    if (width>98 & width<160 & hight>25 & hight<50)%��ܵĿ�Ⱥ͸߶ȵķ�Χ
        lx=lx+1;
        Getok(lx)=l;
    end
end
for k= 1:lx
    l=Getok(k);    
    startcol=BoundingBox((l-1)*4+1)-2;%��ʼ��
    startrow=BoundingBox((l-1)*4+2)-2;%��ʼ��
    width=BoundingBox((l-1)*4+3)+8;%���ƿ�
    hight=BoundingBox((l-1)*4+4)+2;%���Ƹ�
    rato=width/hight;%���㳵�Ƴ����
    if rato>2 & rato<4   
        break;
    end
end
sbw1=bw2(startrow:startrow+hight,startcol:startcol+width-1); %��ȡ���ƶ�ֵ��ͼ
subcol1=Sgray(startrow:startrow+hight,startcol:startcol+width-1);%��ȡ���ƻҶ���ͼ
figure,subplot(2,1,1),imshow(subcol1);title('���ƻҶ���ͼ');%����Ҷ�ͼ��
subplot(2,1,2),imshow(sbw1);title('���ƶ�ֵ��ͼ');%������ƵĶ�ֵͼ
%Step6 ���㳵��ˮƽͶӰ������ˮƽͶӰ���з�ȷ���
histcol1=sum(sbw1);      %���㴹ֱͶӰ
histrow=sum(sbw1');      %����ˮƽͶӰ
figure,subplot(2,1,1),bar(histcol1);title('��ֱͶӰ�����߿�');%�����ֱͶӰ
subplot(2,1,2),bar(histrow);     title('ˮƽͶӰ�����߿�');%���ˮƽͶӰ
figure,subplot(2,1,1),bar(histrow);     title('ˮƽͶӰ�����߿�');%���ˮƽͶӰ
subplot(2,1,2),imshow(sbw1);title('���ƶ�ֵ��ͼ');%�����ֵͼ
%��ˮƽͶӰ���з�ȷ���
meanrow=mean(histrow);%��ˮƽͶӰ��ƽ��ֵ
minrow=min(histrow);%��ˮƽͶӰ����Сֵ
levelrow=(meanrow+minrow)/2;%��ˮƽͶӰ��ƽ��ֵ
count1=0;
l=1;
for k=1:hight
    if histrow(k)<=levelrow                             
        count1=count1+1;                                
    else 
        if count1>=1
            markrow(l)=k;%������
            markrow1(l)=count1;%�ȿ�ȣ��½�������һ�������㣩
            l=l+1;
        end
        count1=0;
    end
end
markrow2=diff(markrow);%����루����������һ�������㣩
[m1,n1]=size(markrow2);
n1=n1+1;
markrow(l)=hight;
markrow1(l)=count1;
markrow2(n1)=markrow(l)-markrow(l-1);
l=0;
for k=1:n1
    markrow3(k)=markrow(k+1)-markrow1(k+1);%�½���
    markrow4(k)=markrow3(k)-markrow(k);%���ȣ����������½��㣩
    markrow5(k)=markrow3(k)-double(uint16(markrow4(k)/2));%������λ��
end 
%Step7 ���㳵����ת�Ƕ�
%(1)�����������½����ҵ�һ��Ϊ1�ĵ�
[m2,n2]=size(sbw1);%sbw1��ͼ���С
[m1,n1]=size(markrow4);%markrow4�Ĵ�С
maxw=max(markrow4);%�����Ϊ�ַ�
if markrow4(1) ~= maxw%����ϱ�
    ysite=1;
    k1=1;
    for l=1:n2
    for k=1:markrow3(ysite)%�Ӷ�������һ�����½���ɨ��
        if sbw1(k,l)==1
            xdata(k1)=l;
            ydata(k1)=k;
            k1=k1+1;
            break;
        end
    end
    end
else  %����±�
    ysite=n1;
    if markrow4(n1) ==0
        if markrow4(n1-1) ==maxw
           ysite= 0; %���±�
       else
           ysite= n1-1;
       end
    end
    if ysite ~=0
        k1=1;
        for l=1:n2
            k=m2;
            while k>=markrow(ysite) %�ӵױ������һ�����������ɨ��
                if sbw1(k,l)==1
                    xdata(k1)=l;
                    ydata(k1)=k;
                    k1=k1+1;
                    break;
                end
                k=k-1;
            end
        end
    end
end       
%(2)������ϣ�������x�н�
fresult = fit(xdata',ydata','poly1');   %poly1    Y = p1*x+p2
p1=fresult.p1;
angle=atan(fresult.p1)*180/pi; %���Ȼ�Ϊ�ȣ�360/2pi,  pi=3.14
%(3)��ת����ͼ��
subcol = imrotate(subcol1,angle,'bilinear','crop'); %��ת����ͼ��
sbw = imrotate(sbw1,angle,'bilinear','crop');%��תͼ��
figure,subplot(2,1,1),imshow(subcol);title('���ƻҶ���ͼ');%���������ת��ĻҶ�ͼ�������ʾ���ƻҶ���ͼ
subplot(2,1,2),imshow(sbw);title('');%���������ת��ĻҶ�ͼ��
title(['������ת��: ',num2str(angle),'��'] ,'Color','r');%��ʾ���Ƶ���ת�Ƕ�
%Step8 ��ת���ƺ����¼��㳵��ˮƽͶӰ��ȥ������ˮƽ�߿򣬻�ȡ�ַ��߶�
histcol1=sum(sbw); %���㴹ֱͶӰ
histrow=sum(sbw'); %����ˮƽͶӰ
figure,subplot(2,1,1),bar(histcol1);title('��ֱͶӰ����ת��');
subplot(2,1,2),bar(histrow);     title('ˮƽͶӰ����ת��');
figure,subplot(2,1,1),bar(histrow);     title('ˮƽͶӰ����ת��');
subplot(2,1,2),imshow(sbw);title('���ƶ�ֵ��ͼ����ת��');
%ȥˮƽ�����£��߿�,��ȡ�ַ��߶�
maxhight=max(markrow2);
findc=find(markrow2==maxhight);
rowtop=markrow(findc);
rowbot=markrow(findc+1)-markrow1(findc+1);
sbw2=sbw(rowtop:rowbot,:);  %��ͼΪ(rowbot-rowtop+1)��
maxhight=rowbot-rowtop+1;   %�ַ��߶�(rowbot-rowtop+1)
%Step9 ���㳵�ƴ�ֱͶӰ��ȥ�����ƴ�ֱ�߿򣬻�ȡ���Ƽ��ַ�ƽ�����
histcol=sum(sbw2);  %���㴹ֱͶӰ
figure,subplot(2,1,1),bar(histcol);title('��ֱͶӰ��ȥˮƽ�߿��');%������ƵĴ�ֱͶӰͼ��
subplot(2,1,2),imshow(sbw2); %�����ֱͶӰͼ��
title(['�����ַ��߶ȣ� ',int2str(maxhight)],'Color','r');%��������ַ��߶�
%�Դ�ֱͶӰ���з�ȷ���
meancol=mean(histcol);%��ֱͶӰ��ƽ��ֵ
mincol=min(histcol);%��ֱͶӰ��ƽ��ֵ
levelcol=(meancol+mincol)/4;%��ֱͶӰ��1/4
count1=0;
l=1;
for k=1:width
    if histcol(k)<=levelcol 
        count1=count1+1;
    else 
        if count1>=1
            markcol(l)=k; %�ַ�������
            markcol1(l)=count1; %�ȿ�ȣ��½�������һ�������㣩
            l=l+1;
        end
        count1=0;
    end
end
markcol2=diff(markcol);%�ַ����루����������һ�������㣩
[m1,n1]=size(markcol2);
n1=n1+1;
markcol(l)=width;
markcol1(l)=count1;
markcol2(n1)=markcol(l)-markcol(l-1);
%Step10 ���㳵����ÿ���ַ�����λ�ã���������ַ����maxwidth
l=0;
for k=1:n1
    markcol3(k)=markcol(k+1)-markcol1(k+1);%�ַ��½���
    markcol4(k)=markcol3(k)-markcol(k); %�ַ���ȣ����������½��㣩
    markcol5(k)=markcol3(k)-double(uint16(markcol4(k)/2));%�ַ�����λ��
end 
markcol6=diff(markcol5); %�ַ����ľ��루�ַ����ĵ�����һ���ַ����ĵ㣩
maxs=max(markcol6); %�������ֵ����Ϊ�ڶ��ַ�������ַ����ľ���
findmax=find(markcol6==maxs);
markcol6(findmax)=0;
maxwidth=max(markcol6);%�������ֵ����Ϊ����ַ����
%Step11 ��ȡ�ָ��ַ�,���任Ϊ22��*14�б�׼��ͼ
l=1;
[m2,n2]=size(subcol);
figure;
for k=findmax-1:findmax+5
        cleft=markcol5(k)-maxwidth/2;
        cright=markcol5(k)+maxwidth/2-2;
        if cleft<1
            cleft=1;
            cright=maxwidth;
        end
        if cright>n2
            cright=n2;
            cleft=n2-maxwidth;
        end
        SegGray=sbw(rowtop:rowbot,cleft:cright);
        SegBw1=sbw(rowtop:rowbot,cleft:cright);
        SegBw2 = imresize(SegBw1,[22 14]);%�任Ϊ22��*14�б�׼��ͼ      
        subplot(2,n1,l),imshow(SegGray);
        if l==7
            title(['�����ַ���ȣ� ',int2str(maxwidth)],'Color','r');
        end
        subplot(2,n1,n1+l),imshow(SegBw2);               
        fname=strcat('H:\work\sam\image',int2str(k),'.jpg');
        imwrite(SegBw2,fname,'jpg') 
        l=l+1;
end
%Step12 ����������ȡ���ַ�ͼ�������������ƥ�䣬�Զ�ʶ����ַ����롣
liccode=char(['0':'9' 'A':'Z' '�����ƹ󴨾���']); %�����Զ�ʶ���ַ������  
SubBw2=zeros(22,14);
l=1;
[m2,n2]=size(sbw);
for k=findmax-1:findmax+5
       cleft=markcol5(k)-maxwidth/2;
        cright=markcol5(k)+maxwidth/2-2;
        if cleft<1
            cleft=1;
            cright=maxwidth;
        end
        if cright>n2
            cright=n2;
            cleft=n2-maxwidth;
        end
        SegBw1=sbw(rowtop:rowbot,cleft:cright);
        SegBw2 = imresize(SegBw1,[22 14]);%�任Ϊ22��*14�б�׼��ͼ      
        if l==1                 %��һλ����ʶ��
            kmin=37;
            kmax=45;
        elseif l==2             %�ڶ�λ A~Z ��ĸʶ��
            kmin=11;
            kmax=36;
        elseif l>=3 & l<=5      %��������λ 0~9  A~Z��ĸ������ʶ��
            kmin=1;
            kmax=36;
        else                    %���填��λ 0~9 ����ʶ��
            kmin=1;
            kmax=10;
        end
        for k2=kmin:kmax
            fname=strcat('H:\work\sam\Sam',liccode(k2),'.jpg');
            SamBw2 = imread(fname);           
            for  i=1:22
                for j=1:14
                    SubBw2(i,j)=SegBw2(i,j)-SamBw2(i,j);
                end
            end %SubBw2 = SamBw2-SegBw2;
            Dmax=0;
            for k1=1:22
                for l1=1:14
                    if ( SubBw2(k1,l1) > 0 | SubBw2(k1,l1) <0 )
                        Dmax=Dmax+1;
                    end
                end
            end
            Error(k2)=Dmax;
        end
        Error1=Error(kmin:kmax);%�Ƚ����
        MinError=min(Error1);%ȡ������Сֵ
        findc=find(Error1==MinError);%������С����ͼ��
        RegCode(l*2-1)=liccode(findc(1)+kmin-1);
        RegCode(l*2)=' ';%�����С���ͼ��
        l=l+1;
end
title (['ʶ���ƺ���:', RegCode],'Color','r');