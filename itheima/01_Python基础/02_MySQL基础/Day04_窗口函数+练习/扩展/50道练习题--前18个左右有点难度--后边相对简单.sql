# 数据表介绍
#
# --1.学生表
# Student(SId,Sname,Sage,Ssex)
# --SId 学生编号,Sname 学生姓名,Sage 出生年月,Ssex 学生性别
#
# --2.课程表
# Course(CId,Cname,TId)
# --CId 课程编号,Cname 课程名称,TId 教师编号
#
# --3.教师表
# Teacher(TId,Tname)
# --TId 教师编号,Tname 教师姓名
#
# --4.成绩表
# SC(SId,CId,score)
# --SId 学生编号,CId 课程编号,score 分数
#
# 学生表 Student
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
values ('01', '赵雷', '1990-01-01', '男');
insert into Student
values ('02', '钱电', '1990-12-21', '男');
insert into Student
values ('03', '孙风', '1990-12-20', '男');
insert into Student
values ('04', '李云', '1990-12-06', '男');
insert into Student
values ('05', '周梅', '1991-12-01', '女');
insert into Student
values ('06', '吴兰', '1992-01-01', '女');
insert into Student
values ('07', '郑竹', '1989-01-01', '女');
insert into Student
values ('09', '张三', '2017-12-20', '女');
insert into Student
values ('10', '李四', '2017-12-25', '女');
insert into Student
values ('11', '李四', '2012-06-06', '女');
insert into Student
values ('12', '赵六', '2013-06-13', '女');
insert into Student
values ('13', '孙七', '2014-06-01', '女');
# 科目表 Course
create table Course
(
    CId   varchar(10),
    Cname nvarchar(10),
    TId   varchar(10)
);
insert into Course
values ('01', '语文', '02');
insert into Course
values ('02', '数学', '01');
insert into Course
values ('03', '英语', '03');
# 教师表 Teacher
create table Teacher
(
    TId   varchar(10),
    Tname varchar(10)
);
insert into Teacher
values ('01', '张三');
insert into Teacher
values ('02', '李四');
insert into Teacher
values ('03', '王五');
# 成绩表 SC
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
-- 1.查询" 01 "课程比" 02 "课程成绩高的学生的信息及课程分数
select t3.*, t1.CId, t1.score, t2.CId, t2.score
from (select * from sc where CId = 01) t1,
     (select * from sc where CId = 02) t2,
     student t3
where t1.SId = t2.SId
  and t3.SId = t1.SId
  and t1.score > t2.score
;

-- 1.1 查询同时存在" 01 "课程和" 02 "课程的情况
select *
from (select * from sc where CId = 01) t1,
     (select * from sc where CId = 02) t2
where t1.SId = t2.SId;

-- 1.2 查询存在" 01 "课程但可能不存在" 02 "课程的情况(不存在时显示为 null )
select *
from (select * from sc where CId = 01) t1
         left join
         (select * from sc where CId = 02) t2
         on t1.SId = t2.SId
where t2.SId is null;
-- 1.3 查询不存在" 01 "课程但存在" 02 "课程的情况
select *
from (select * from sc where CId = 01) t1
         right join
         (select * from sc where CId = 02) t2
         on t1.SId = t2.SId
where t1.SId is null;


-- 2.查询平均成绩大于等于 60 分的同学的学生编号和学生姓名和平均成绩
select t1.SId, Sname, avg_score
from student,
     (select SId, avg(score) avg_score from sc group by SId having avg_score >= 60) t1
where Student.SId = t1.SId


-- 3.查询在 SC 表存在成绩的学生信息
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

-- 4.查询所有同学的学生编号、学生姓名、选课总数、所有课程的成绩总和

-- 4.1显示没选课的学生(显示为NULL)
select Student.sid,
       Student.Sname,
       count(CId) cnt,
#        if(sum(score) is null, 0, sum(score)) total_score
#       case when count(cid) = 0 then null else count(cid) end cnt
#        nullif(count(cid), "null") cnt
from student
         left join sc on Student.SId = SC.SId
group by Student.SId, Sname;

-- 4.2查有成绩的学生信息
select s.SId, Sname, count(cid) cnt, sum(score) total_score
from student s
         join sc on s.SId = SC.SId
group by s.SId, Sname

-- 5.查询「李」姓老师的数量
select count(1) cnt
from teacher
where Tname = "李%"

help nullif;
help year
select year("1999-01-01")


-- 6.查询学过「张三」老师授课的同学的信息
select *
from student
where SId in (select SId
              from sc
              where CId in (select CId
                            from course
                            where TId in (select TId from teacher where Tname = "张三")))


-- 7.查询没有学全所有课程的同学的信息
select *
from student,
     (select SId, count(1) cnt
      from sc
      group by SId
      having cnt !=
             (select count(1) cnt from course)) t1
where Student.SId = t1.SId
-- 8.查询至少有一门课与学号为" 01 "的同学所学相同的同学的信息

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


-- 9.查询和" 01 "号的同学学习的课程完全相同的其他同学的信息
-- 解法一
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


-- 解法二
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


-- 10.查询没学过"张三"老师讲授的任一门课程的学生姓名
select CId
from teacher t,
     course c
where Tname = "张三"
  and c.TId = t.TId
select *
from student
where SId not in (select SId
                  from sc
                  where CId in (select CId
                                from teacher t,
                                     course c
                                where Tname = "张三"
                                  and c.TId = t.TId))

-- 11.查询两门及其以上不及格课程的同学的学号，姓名及其平均成绩

select SC.SId, Sname, avg(score) avg_score
from sc,
     student
where score < 60
  and SC.SId = Student.SId
group by SId, Sname
having count(CId) >= 2


-- 12.检索" 01 "课程分数小于 60，按分数降序排列的学生信息
select Student.*
from sc,
     student
where score < 60
  and CId = 01
  and SC.SId = Student.SId

-- 13.按平均成绩从高到低显示所有学生的所有课程的成绩以及平均成绩
select *
from sc,
     (select SId, avg(score) avg_s from sc group by SId) t1
where SC.SId = t1.SId
order by avg_s desc


-- 14.查询各科成绩最高分、最低分和平均分
-- 以如下形式显示：课程 ID，课程 name，最高分，最低分，平均分，及格率，中等率，优良率，优秀率
-- 及格为>=60，中等为：70-80，优良为：80-90，优秀为：>=90
-- 要求输出课程号和选修人数，查询结果按人数降序排列，若人数相同，按课程号升序排列
select c.CId,
       count(1)                                                                          cnt,
       max(score)                                                                        "最高分",
       min(score)                                                                        "最低分",
       avg(score)                                                                        "平均分",
       concat(round(count(score >= 60 or null) / count(1) * 100, 2), "%")                "及格率",
       concat(round(count(score >= 70 and score < 80 or null) / count(1) * 100, 2), "%") "中等率",
       concat(round(count(score >= 80 and score < 90 or null) / count(1) * 100, 2), "%") "优良率",
       concat(round(count(score >= 90 or null) / count(1) * 100), 2, "%")                "优秀率"
from sc,
     course c
where c.CId = SC.CId
group by c.CId, Cname
order by cnt desc, CId

select concat("a", "b")

-- 15.按各科成绩进行排序，并显示排名， Score 重复时保留名次空缺
select *, rank() over (partition by cid order by score desc) rk
from sc;


-- 16.查询学生的总成绩，并进行排名，总分重复时不保留名次空缺
select *, dense_rank() over (order by total_score desc) dr
from (select sid, sum(score) total_score from sc group by sid) t1


-- 17.统计各科成绩各分数段人数：课程编号，课程名称，[100-85]，[85-70]，[70-60]，[60-0] 及所占百分比
select c.CId, Cname
from course c,
     sc
where c.CId = SC.CId
group by c.CId, Cname

-- 18. 查询各科成绩前三名的记录

-- 19.查询每门课程被选修的学生数


-- 20.查询出只选修两门课程的学生学号和姓名

-- 21.查询男生、女生人数
select Ssex, count(SId) num
from student
group by Ssex;

-- 22.查询名字中含有「风」字的学生信息
select *
from student
where Sname like "%风%";
-- 23.查询同名学生名单，并统计同名人数
select Sname, cnt
from (select Sname, count(SId) cnt from student group by Sname) t1
where cnt > 1;


-- 24.查询 1990 年出生的学生名单
select Sname, Sage
from student
where Sage >= "1990-01-01"
  and Sage <= "1990-12-31";

-- 25.查询每门课程的平均成绩，结果按平均成绩降序排列，平均成绩相同时，按课程编号升序排列
select CId, round(avg(score), 2) avg_sc
from sc
group by CId
order by avg_sc desc, CId asc;

-- 26.查询平均成绩大于等于 85 的所有学生的学号、姓名和平均成绩
select Student.SId, Sname, avg_sc
from student
         inner join (select SId, round(avg(score), 2) avg_sc from sc group by SId having avg_sc > 85) t1
where Student.SId = t1.SId;


-- 27.查询课程名称为「数学」，且分数低于 60
select *
from course
         inner join (select *
                     from sc
                     where score < 60) t1
                    on Course.CId = t1.CId
where Cname = "数学";
;
-- 28.查询所有学生的课程及分数情况（存在学生没成绩，没选课的情况）
select Sname, CId, score
from student
         left join sc on Student.SId = SC.SId;

-- 29.查询任何一门课程成绩在 70 分以上的姓名、课程名称和分数
-- 理解一是任意一门成绩均在70分以上
-- 理解二是存在一门成绩在70分以上即可满足条件
-- 理解三就是找出所有大于70分的得分。


-- 30.查询存在不及格的课程
select distinct (CId)
from sc
where score < 60;
-- 31.查询课程编号为 01 且课程成绩在 80 分及以上的学生的学号和姓名
select Student.SId, Sname
from student
         inner join (select SId
                     from sc
                     where score >= 80
                       and CId = 01) t1
where Student.SId = t1.SId

-- 32.求每门课程的学生人数
select CId, count(SId) num
from sc
group by CId;
-- 33.成绩不重复，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
select SId, score
from sc,
     course,
     teacher
where SC.CId = Course.CId
  and Course.TId = Teacher.TId
  and Tname = "张三"
order by score desc
limit 1;

-- 34.成绩有重复的情况下，查询选修「张三」老师所授课程的学生中，成绩最高的学生信息及其成绩
select t1.SId, t1.score
from (select SId, score
      from sc,
           course,
           teacher
      where SC.CId = Course.CId
        and Course.TId = Teacher.TId
        and Tname = "张三"
      order by score desc) t2
         inner join
     (select SId, score
      from sc,
           course,
           teacher
      where SC.CId = Course.CId
        and Course.TId = Teacher.TId
        and Tname = "张三"
      order by score desc
      limit 1) t1 on t1.score = t2.score;


-- 35.查询不同课程成绩相同的学生的学生编号、课程编号、学生成绩
select distinct s1.SId, s1.CId, s1.score
from sc s1
         left join sc s2 on s1.SId = s2.SId
where s1.CId != s2.CId
  and s1.score = s2.score;

-- 这个问题其实一开始没太明白啥意思，后来理解为某个人的几科分数是一样的，需要把这个人找出来

-- 36.查询每门功成绩最好的前两名

SELECT STU.*,
       S.score,
       C.Cname,
       ROW_NUMBER() over (PARTITION BY S.CID ORDER BY S.score DESC) 排名
FROM student STU
         INNER JOIN SC S on STU.SID = S.SID
         INNER JOIN Course C on S.CID = C.CID;

SELECT *
FROM (SELECT STU.*,
             S.score,
             C.Cname,
             ROW_NUMBER() over (PARTITION BY S.CID ORDER BY S.score DESC) 排名
      FROM student STU
               INNER JOIN SC S on STU.SID = S.SID
               INNER JOIN Course C on S.CID = C.CID) T
WHERE T.排名 <= 2;

-- 37.统计每门课程的学生选修人数（超过 5 人的课程才统计）
select CId, count(SId) num
from sc
group by CId
having num > 5
order by num desc, CId asc
;
-- 38.检索至少选修两门课程的学生学号
select SId, count(CId) num
from sc
group by SId
having num >= 2;
-- 39.查询选修了全部课程的学生信息
select count(CId)
from course;



-- 40.查询各学生的年龄，只按年份来算

-- 41.按照出生日期来算，当前月日 < 出生年月的月日则，年龄减一

-- 42.查询本周过生日的学生
-- 有点复杂，需要拼接出本周的起止日期

-- 43. 查询下周过生日的学生
-- 同42

-- 44.查询本月过生日的学生

-- 45.查询下月过生日的学生
-- 注意本月是12月的话，下一个月份是1即可


