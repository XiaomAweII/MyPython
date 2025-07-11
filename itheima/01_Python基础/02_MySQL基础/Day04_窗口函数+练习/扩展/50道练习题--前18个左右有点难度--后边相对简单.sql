# ���ݱ����
#
# --1.ѧ����
# Student(SId,Sname,Sage,Ssex)
# --SId ѧ�����,Sname ѧ������,Sage ��������,Ssex ѧ���Ա�
#
# --2.�γ̱�
# Course(CId,Cname,TId)
# --CId �γ̱��,Cname �γ�����,TId ��ʦ���
#
# --3.��ʦ��
# Teacher(TId,Tname)
# --TId ��ʦ���,Tname ��ʦ����
#
# --4.�ɼ���
# SC(SId,CId,score)
# --SId ѧ�����,CId �γ̱��,score ����
#
# ѧ���� Student
create database exercise;
use exercise;
create table Student
(
    SId   varchar(10),
    Sname varchar(10),
    Sage  datetime,
    Ssex  varchar(10)
);
insert into Student
values ('01', '����', '1990-01-01', '��');
insert into Student
values ('02', 'Ǯ��', '1990-12-21', '��');
insert into Student
values ('03', '���', '1990-12-20', '��');
insert into Student
values ('04', '����', '1990-12-06', '��');
insert into Student
values ('05', '��÷', '1991-12-01', 'Ů');
insert into Student
values ('06', '����', '1992-01-01', 'Ů');
insert into Student
values ('07', '֣��', '1989-01-01', 'Ů');
insert into Student
values ('09', '����', '2017-12-20', 'Ů');
insert into Student
values ('10', '����', '2017-12-25', 'Ů');
insert into Student
values ('11', '����', '2012-06-06', 'Ů');
insert into Student
values ('12', '����', '2013-06-13', 'Ů');
insert into Student
values ('13', '����', '2014-06-01', 'Ů');
# ��Ŀ�� Course
create table Course
(
    CId   varchar(10),
    Cname nvarchar(10),
    TId   varchar(10)
);
insert into Course
values ('01', '����', '02');
insert into Course
values ('02', '��ѧ', '01');
insert into Course
values ('03', 'Ӣ��', '03');
# ��ʦ�� Teacher
create table Teacher
(
    TId   varchar(10),
    Tname varchar(10)
);
insert into Teacher
values ('01', '����');
insert into Teacher
values ('02', '����');
insert into Teacher
values ('03', '����');
# �ɼ��� SC
create table SC
(
    SId   varchar(10),
    CId   varchar(10),
    score decimal(18, 1)
);
insert into SC
values ('01', '01', 80);
insert into SC
values ('01', '02', 90);
insert into SC
values ('01', '03', 99);
insert into SC
values ('02', '01', 70);
insert into SC
values ('02', '02', 60);
insert into SC
values ('02', '03', 80);
insert into SC
values ('03', '01', 80);
insert into SC
values ('03', '02', 80);
insert into SC
values ('03', '03', 80);
insert into SC
values ('04', '01', 50);
insert into SC
values ('04', '02', 30);
insert into SC
values ('04', '03', 20);
insert into SC
values ('05', '01', 76);
insert into SC
values ('05', '02', 87);
insert into SC
values ('06', '01', 31);
insert into SC
values ('06', '03', 34);
insert into SC
values ('07', '02', 89);
insert into SC
values ('07', '03', 98);

show tables;
-- 1.��ѯ" 01 "�γ̱�" 02 "�γ̳ɼ��ߵ�ѧ������Ϣ���γ̷���
select t3.*, t1.CId, t1.score, t2.CId, t2.score
from (select * from sc where CId = 01) t1,
     (select * from sc where CId = 02) t2,
     student t3
where t1.SId = t2.SId
  and t3.SId = t1.SId
  and t1.score > t2.score
;

-- 1.1 ��ѯͬʱ����" 01 "�γ̺�" 02 "�γ̵����
select *
from (select * from sc where CId = 01) t1,
     (select * from sc where CId = 02) t2
where t1.SId = t2.SId;

-- 1.2 ��ѯ����" 01 "�γ̵����ܲ�����" 02 "�γ̵����(������ʱ��ʾΪ null )
select *
from (select * from sc where CId = 01) t1
         left join
         (select * from sc where CId = 02) t2
         on t1.SId = t2.SId
where t2.SId is null;
-- 1.3 ��ѯ������" 01 "�γ̵�����" 02 "�γ̵����
select *
from (select * from sc where CId = 01) t1
         right join
         (select * from sc where CId = 02) t2
         on t1.SId = t2.SId
where t1.SId is null;


-- 2.��ѯƽ���ɼ����ڵ��� 60 �ֵ�ͬѧ��ѧ����ź�ѧ��������ƽ���ɼ�
select t1.SId, Sname, avg_score
from student,
     (select SId, avg(score) avg_score from sc group by SId having avg_score >= 60) t1
where Student.SId = t1.SId


-- 3.��ѯ�� SC ����ڳɼ���ѧ����Ϣ
select *
from sc,
     student
where SC.SId = Student.SId
select distinct sid
from sc
select *
from student
where SId in (select distinct sid from sc);
select Student.*
from student,
     (select distinct SC.SId from sc) t1
where Student.SId = t1.SId

select distinct *
from student t1,
     (select sid from sc) t2
where t1.SId = t2.SId

-- 4.��ѯ����ͬѧ��ѧ����š�ѧ��������ѡ�����������пγ̵ĳɼ��ܺ�

-- 4.1��ʾûѡ�ε�ѧ��(��ʾΪNULL)
select Student.sid,
       Student.Sname,
       count(CId) cnt,
#        if(sum(score) is null, 0, sum(score)) total_score
#       case when count(cid) = 0 then null else count(cid) end cnt
#        nullif(count(cid), "null") cnt
from student
         left join sc on Student.SId = SC.SId
group by Student.SId, Sname;

-- 4.2���гɼ���ѧ����Ϣ
select s.SId, Sname, count(cid) cnt, sum(score) total_score
from student s
         join sc on s.SId = SC.SId
group by s.SId, Sname

-- 5.��ѯ�������ʦ������
select count(1) cnt
from teacher
where Tname = "��%"

help nullif;
help year
select year("1999-01-01")


-- 6.��ѯѧ������������ʦ�ڿε�ͬѧ����Ϣ
select *
from student
where SId in (select SId
              from sc
              where CId in (select CId
                            from course
                            where TId in (select TId from teacher where Tname = "����")))


-- 7.��ѯû��ѧȫ���пγ̵�ͬѧ����Ϣ
select *
from student,
     (select SId, count(1) cnt
      from sc
      group by SId
      having cnt !=
             (select count(1) cnt from course)) t1
where Student.SId = t1.SId
-- 8.��ѯ������һ�ſ���ѧ��Ϊ" 01 "��ͬѧ��ѧ��ͬ��ͬѧ����Ϣ

select distinct SId
from sc
where CId in (select CId from sc where SId = 01)
  and SId != 01
select cid
from sc
where SId = 01


select *
from student
where sid in (select distinct sid from sc where CId in (select cid from sc where SId = 01) and SId != 01);

select *
from student s,
     (select distinct sid
      from sc
      where CId in (select cid from sc where SId = 01)) t1
where s.sid = t1.SId
  and s.SId != 01


-- 9.��ѯ��" 01 "�ŵ�ͬѧѧϰ�Ŀγ���ȫ��ͬ������ͬѧ����Ϣ
-- �ⷨһ
select sid, count(1) cnt
from sc
where SC.SId != 01
group by sid
select count(CId) cnt
from SC
where SId = 01
select sid
from sc
where SId != 01
group by sid
having count(1) = (select count(CId) cnt from SC where SId = 01)

select s.*
from (select count(1) cnt from sc where sid = 01) t1,
     (select sid, count(1) cnt from sc where sid != 01 group by sid) t2,
     student s
where t1.cnt = t2.cnt
  and s.SId = t2.SId


-- �ⷨ��
help group_concat;
select group_concat(score)
from sc
select group_concat(score order by score)
from sc

select group_concat(cid order by cid) gc
from sc
where SId = 01

select Student.*
from (select sid, group_concat(cid order by cid) gc from sc group by SId) t1,
     (select group_concat(cid order by cid) gc from sc where SId = 01) t2,
     student
where t1.gc = t2.gc
  and Student.SId = t1.SId


-- 10.��ѯûѧ��"����"��ʦ���ڵ���һ�ſγ̵�ѧ������
select CId
from teacher t,
     course c
where Tname = "����"
  and c.TId = t.TId
select *
from student
where SId not in (select SId
                  from sc
                  where CId in (select CId
                                from teacher t,
                                     course c
                                where Tname = "����"
                                  and c.TId = t.TId))

-- 11.��ѯ���ż������ϲ�����γ̵�ͬѧ��ѧ�ţ���������ƽ���ɼ�

select SC.SId, Sname, avg(score) avg_score
from sc,
     student
where score < 60
  and SC.SId = Student.SId
group by SId, Sname
having count(CId) >= 2


-- 12.����" 01 "�γ̷���С�� 60���������������е�ѧ����Ϣ
select Student.*
from sc,
     student
where score < 60
  and CId = 01
  and SC.SId = Student.SId

-- 13.��ƽ���ɼ��Ӹߵ�����ʾ����ѧ�������пγ̵ĳɼ��Լ�ƽ���ɼ�
select *
from sc,
     (select SId, avg(score) avg_s from sc group by SId) t1
where SC.SId = t1.SId
order by avg_s desc


-- 14.��ѯ���Ƴɼ���߷֡���ͷֺ�ƽ����
-- ��������ʽ��ʾ���γ� ID���γ� name����߷֣���ͷ֣�ƽ���֣������ʣ��е��ʣ������ʣ�������
-- ����Ϊ>=60���е�Ϊ��70-80������Ϊ��80-90������Ϊ��>=90
-- Ҫ������γ̺ź�ѡ����������ѯ����������������У���������ͬ�����γ̺���������
select c.CId,
       count(1)                                                                          cnt,
       max(score)                                                                        "��߷�",
       min(score)                                                                        "��ͷ�",
       avg(score)                                                                        "ƽ����",
       concat(round(count(score >= 60 or null) / count(1) * 100, 2), "%")                "������",
       concat(round(count(score >= 70 and score < 80 or null) / count(1) * 100, 2), "%") "�е���",
       concat(round(count(score >= 80 and score < 90 or null) / count(1) * 100, 2), "%") "������",
       concat(round(count(score >= 90 or null) / count(1) * 100), 2, "%")                "������"
from sc,
     course c
where c.CId = SC.CId
group by c.CId, Cname
order by cnt desc, CId

select concat("a", "b")

-- 15.�����Ƴɼ��������򣬲���ʾ������ Score �ظ�ʱ�������ο�ȱ
select *, rank() over (partition by cid order by score desc) rk
from sc;


-- 16.��ѯѧ�����ܳɼ����������������ܷ��ظ�ʱ���������ο�ȱ
select *, dense_rank() over (order by total_score desc) dr
from (select sid, sum(score) total_score from sc group by sid) t1


-- 17.ͳ�Ƹ��Ƴɼ����������������γ̱�ţ��γ����ƣ�[100-85]��[85-70]��[70-60]��[60-0] ����ռ�ٷֱ�
select c.CId, Cname
from course c,
     sc
where c.CId = SC.CId
group by c.CId, Cname

-- 18. ��ѯ���Ƴɼ�ǰ�����ļ�¼

-- 19.��ѯÿ�ſγ̱�ѡ�޵�ѧ����


-- 20.��ѯ��ֻѡ�����ſγ̵�ѧ��ѧ�ź�����

-- 21.��ѯ������Ů������
select Ssex, count(SId) num
from student
group by Ssex;

-- 22.��ѯ�����к��С��硹�ֵ�ѧ����Ϣ
select *
from student
where Sname like "%��%";
-- 23.��ѯͬ��ѧ����������ͳ��ͬ������
select Sname, cnt
from (select Sname, count(SId) cnt from student group by Sname) t1
where cnt > 1;


-- 24.��ѯ 1990 �������ѧ������
select Sname, Sage
from student
where Sage >= "1990-01-01"
  and Sage <= "1990-12-31";

-- 25.��ѯÿ�ſγ̵�ƽ���ɼ��������ƽ���ɼ��������У�ƽ���ɼ���ͬʱ�����γ̱����������
select CId, round(avg(score), 2) avg_sc
from sc
group by CId
order by avg_sc desc, CId asc;

-- 26.��ѯƽ���ɼ����ڵ��� 85 ������ѧ����ѧ�š�������ƽ���ɼ�
select Student.SId, Sname, avg_sc
from student
         inner join (select SId, round(avg(score), 2) avg_sc from sc group by SId having avg_sc > 85) t1
where Student.SId = t1.SId;


-- 27.��ѯ�γ�����Ϊ����ѧ�����ҷ������� 60
select *
from course
         inner join (select *
                     from sc
                     where score < 60) t1
                    on Course.CId = t1.CId
where Cname = "��ѧ";
;
-- 28.��ѯ����ѧ���Ŀγ̼��������������ѧ��û�ɼ���ûѡ�ε������
select Sname, CId, score
from student
         left join sc on Student.SId = SC.SId;

-- 29.��ѯ�κ�һ�ſγ̳ɼ��� 70 �����ϵ��������γ����ƺͷ���
-- ���һ������һ�ųɼ�����70������
-- �����Ǵ���һ�ųɼ���70�����ϼ�����������
-- ����������ҳ����д���70�ֵĵ÷֡�


-- 30.��ѯ���ڲ�����Ŀγ�
select distinct (CId)
from sc
where score < 60;
-- 31.��ѯ�γ̱��Ϊ 01 �ҿγ̳ɼ��� 80 �ּ����ϵ�ѧ����ѧ�ź�����
select Student.SId, Sname
from student
         inner join (select SId
                     from sc
                     where score >= 80
                       and CId = 01) t1
where Student.SId = t1.SId

-- 32.��ÿ�ſγ̵�ѧ������
select CId, count(SId) num
from sc
group by CId;
-- 33.�ɼ����ظ�����ѯѡ�ޡ���������ʦ���ڿγ̵�ѧ���У��ɼ���ߵ�ѧ����Ϣ����ɼ�
select SId, score
from sc,
     course,
     teacher
where SC.CId = Course.CId
  and Course.TId = Teacher.TId
  and Tname = "����"
order by score desc
limit 1;

-- 34.�ɼ����ظ�������£���ѯѡ�ޡ���������ʦ���ڿγ̵�ѧ���У��ɼ���ߵ�ѧ����Ϣ����ɼ�
select t1.SId, t1.score
from (select SId, score
      from sc,
           course,
           teacher
      where SC.CId = Course.CId
        and Course.TId = Teacher.TId
        and Tname = "����"
      order by score desc) t2
         inner join
     (select SId, score
      from sc,
           course,
           teacher
      where SC.CId = Course.CId
        and Course.TId = Teacher.TId
        and Tname = "����"
      order by score desc
      limit 1) t1 on t1.score = t2.score;


-- 35.��ѯ��ͬ�γ̳ɼ���ͬ��ѧ����ѧ����š��γ̱�š�ѧ���ɼ�
select distinct s1.SId, s1.CId, s1.score
from sc s1
         left join sc s2 on s1.SId = s2.SId
where s1.CId != s2.CId
  and s1.score = s2.score;

-- ���������ʵһ��ʼû̫����ɶ��˼���������Ϊĳ���˵ļ��Ʒ�����һ���ģ���Ҫ��������ҳ���

-- 36.��ѯÿ�Ź��ɼ���õ�ǰ����

SELECT STU.*,
       S.score,
       C.Cname,
       ROW_NUMBER() over (PARTITION BY S.CID ORDER BY S.score DESC) ����
FROM student STU
         INNER JOIN SC S on STU.SID = S.SID
         INNER JOIN Course C on S.CID = C.CID;

SELECT *
FROM (SELECT STU.*,
             S.score,
             C.Cname,
             ROW_NUMBER() over (PARTITION BY S.CID ORDER BY S.score DESC) ����
      FROM student STU
               INNER JOIN SC S on STU.SID = S.SID
               INNER JOIN Course C on S.CID = C.CID) T
WHERE T.���� <= 2;

-- 37.ͳ��ÿ�ſγ̵�ѧ��ѡ������������ 5 �˵Ŀγ̲�ͳ�ƣ�
select CId, count(SId) num
from sc
group by CId
having num > 5
order by num desc, CId asc
;
-- 38.��������ѡ�����ſγ̵�ѧ��ѧ��
select SId, count(CId) num
from sc
group by SId
having num >= 2;
-- 39.��ѯѡ����ȫ���γ̵�ѧ����Ϣ
select count(CId)
from course;



-- 40.��ѯ��ѧ�������䣬ֻ���������

-- 41.���ճ����������㣬��ǰ���� < �������µ������������һ

-- 42.��ѯ���ܹ����յ�ѧ��
-- �е㸴�ӣ���Ҫƴ�ӳ����ܵ���ֹ����

-- 43. ��ѯ���ܹ����յ�ѧ��
-- ͬ42

-- 44.��ѯ���¹����յ�ѧ��

-- 45.��ѯ���¹����յ�ѧ��
-- ע�Ȿ����12�µĻ�����һ���·���1����


