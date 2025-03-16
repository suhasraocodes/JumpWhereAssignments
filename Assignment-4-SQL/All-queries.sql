
-- create
CREATE TABLE DEPT (
  deptno int PRIMARY KEY,
  dname VARCHAR(30) ,
  loc VARCHAR(50) 
);

CREATE TABLE EMP (
  Empno int PRIMARY KEY,
  Ename VARCHAR(100) NOT NULL,
  sal int,
  hire_date date NOT NULL,
  commission int NULL,
  deptno int NULL,
  Mgr int ,
  foreign KEY(deptno) references DEPT(deptno) on delete set NULL,
  foreign KEY(Mgr) references EMP(Empno) on delete set NULL
);


INSERT INTO Dept (DeptNo, Dname, Loc) VALUES
(10, 'Accounts', 'Bangalore'),
(20, 'IT', 'Delhi'),
(30, 'Production', 'Chennai'),
(40, 'Sales', 'Hyd'),
(50, 'Admn', 'London');

INSERT INTO Emp (EmpNo, Ename, Sal, Hire_Date, Commission, Deptno, Mgr) VALUES
(1007, 'Martin', 21000, '2000-01-01', 1040, NULL, NULL);
INSERT INTO Emp (EmpNo, Ename, Sal, Hire_Date, Commission, Deptno, Mgr) VALUES
(1006, 'Dravid', 19000, '1985-01-01', 2400, 10, 1007),
(1005, 'John', 5000, '2005-01-01', NULL, 30, 1006),
(1004, 'Williams', 9000, '2001-01-01', NULL, 30, 1007),
(1003, 'Stefen', 12000, '1990-01-01', 500, 20, 1007),
(1002, 'Kapil', 15000, '1970-01-01', 2300, 10, 1003),
(1001, 'Sachin', 19000, '1980-01-01', 2100, 20, 1003);

--1)Select employee details  of dept number 10 or 30
select * from emp where Deptno=10 or Deptno=30;

--2)Write a query to fetch all the dept details with more than 1 Employee.
select * from Dept where
deptno in (select deptno from emp 
          group by deptno
           having count(EmpNo)>1);

--3)Write a query to fetch employee details whose name starts with the letter “S”
select * from Emp
where Ename Like "s%";

--4)Select Emp Details Whose experience is more than 2 years
select * ,timestampdiff(year,Hire_Date,curdate()) as experience from Emp
where timestampdiff(year,Hire_Date,curdate())>2;

--5)Write a SELECT statement to replace the char “a” with “#” in Employee Name ( Ex:  Sachin as S#chin)
select *,replace(ename,'a','#') as modifiedname from emp;

--6)Write a query to fetch employee name and his/her manager name. 
select e1.ename as employee,e2.ename as manager
from emp e1 left join emp e2 
on e1.mgr=e2.empno;

--7)Fetch Dept Name , Total Salry of the Dept
select d.dname as department,sum(e.sal) as salary 
from emp e join dept d 
on e.Deptno=d.Deptno 
group by d.dname;

--8)Write a query to fetch ALL the  employee details along with department name, department location, irrespective of employee existance in the department.
select e.EmpNo,e.Ename,  e.Sal,e.Hire_Date, e.Commission, e.Mgr,d.dname AS Department_Name,d.loc AS Department_Location
       from emp e right join dept d 
       on e.deptno=d.deptno; 


--9)Write an update statement to increase the employee salary by 10 %
select * from emp;

update Emp
set sal=sal*1.1;

select * from emp;

--10)Write a statement to delete employees belong to Chennai location.
select * from emp;

delete from emp
where deptno in(select deptno from  dept where loc="Chennai");

select * from emp;

--11)Get Employee Name and gross salary (sal + comission) .
select ename,(sal+ coalesce(Commission,0))as gross_salary from emp;

--12)Increase the data length of the column Ename of Emp table from  100 to 250 using ALTER statement
desc emp;

alter table emp 
modify column ename VARCHAR(250);

desc emp;

--13)Write query to get current datetime
select now() as current_datetime;

--14)Write a statement to create STUDENT table, with related 5 columns
create table student(
usn varchar(10) primary key,
sname varchar(100) not NULL,
sage int not NULL,
class int ,
section varchar(1)
);
desc student;

--15) Write a query to fetch number of employees in who is getting salary more than 10000
select count(EmpNo) as Employee_count from Emp
where sal>10000;

--16)Write a query to fetch minimum salary, maximum salary and average salary from emp table.
select min(sal) as Minimum_Salary,
   max(sal) as Maximum_Salary,
   avg(sal) as Average_salary from emp;

--17)Write a query to fetch number of employees in each location
select d.loc as location, count(e.empno) as Employee_count
from emp e right join dept d 
on e.Deptno=d.Deptno
group by d.loc;

--18)Write a query to display emplyee names in descending order
select ename from Emp
order by ename Desc;

--19)Write a statement to create a new table(EMP_BKP) from the existing EMP table 
create table EMP_BKP as
select * from emp;
select * from EMP_BKP;

--20) Write a query to fetch first 3 characters from employee name appended with salary.
select concat(left(ename,3),sal) as Employee_info from emp;

--21)Get the details of the employees whose name starts with S
select * from Emp
where ename like "s%";

--22)Get the details of the employees who works in Bangalore location
select * from emp e right join dept d on d.Deptno=e.Deptno
where d.loc="Bangalore";

--23) Write the query to get the employee details whose name started within  any letter between  A and K
select * from Emp
where ename between "A%" and "k%";

--24)Write a query in SQL to display the employees whose manager name is Stefen 
select e1.* from emp e1 join emp e2 
on e1.Mgr =e2.empno
where e2.ename="Stefen";

--25) Write a query in SQL to list the name of the managers who is having maximum number of employees working under him
select e2.ename as Manager_Name, count(e1.EmpNo) as Employee_Count
from emp e1 join emp e2 on e1.Mgr = e2.EmpNo
group by e2.Ename
having count(e1.EmpNo) = (
    select max(Employee_Count)
   from (
        select count(EmpNo) as Employee_Count
        from Emp
        where Mgr is not NULL
        group by Mgr
    ) as ManagerCounts
);

--26) Write a query to display the employee details, department details and the manager details of the employee who has second highest salary
select EmpNo, Ename, Sal, DeptNo, Mgr
from Emp 
order by Sal DESC 
LIMIT 1 OFFSET 1;

--27) Write a query to list all details of all the managers
select e.* from emp e
where e.empno IN 
(select distinct Mgr from emp 
where mgr is not null
);

--28) Write a query to list the details and total experience of all the managers
select e.* ,timestampdiff(year,Hire_Date,curdate()) as Experience from emp e
where e.empno IN (select distinct Mgr from emp where mgr is not null);

--29) Write a query to list the employees who is manager and  takes commission less than 1000 and works in Delhi
select DISTINCT e.*
from Emp e
join Dept d on e.DeptNo = d.DeptNo
where e.EmpNo in (select distinct Mgr from Emp where Mgr is not null)
and e.Commission < 1000
and d.Loc = 'Delhi';

--30) Write a query to display the details of employees who are senior to Martin 
select * from emp 
where Hire_Date<(select Hire_Date from emp where Ename='Martin');