-- Converted query for original CSV row 0
WITH RECURSIVE employee_hierarchy AS (
    SELECT
        employee_id,
        first_name,
        last_name,
        manager_id,
        job_id,
        1 AS hierarchy_level,
        ARRAY[last_name, first_name] AS sort_path
    FROM
        employees
    WHERE
        manager_id IS NULL
    UNION ALL
    SELECT
        e.employee_id,
        e.first_name,
        e.last_name,
        e.manager_id,
        e.job_id,
        eh.hierarchy_level + 1,
        eh.sort_path || ARRAY[e.last_name, e.first_name]
    FROM
        employees e
    JOIN
        employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT
    employee_id,
    LPAD('', (hierarchy_level - 1) * 2) || first_name || ' ' || last_name AS employee_name,
    hierarchy_level,
    manager_id,
    job_id
FROM
    employee_hierarchy
ORDER BY
    sort_path

-- Converted query for original CSV row 1
SELECT
    d.department_name,
    e_ranked.first_name,
    e_ranked.last_name,
    e_ranked.salary,
    e_ranked.rank_in_dept
FROM
    departments d
LEFT JOIN
    (
        SELECT
            e.employee_id,
            e.first_name,
            e.last_name,
            e.salary,
            e.department_id,
            ROW_NUMBER() OVER (PARTITION BY e.department_id ORDER BY e.salary DESC NULLS LAST) as rank_in_dept
        FROM
            employees e
    ) e_ranked
ON
    d.department_id = e_ranked.department_id
WHERE
    e_ranked.rank_in_dept <= 2 OR e_ranked.rank_in_dept IS NULL
ORDER BY
    d.department_name, e_ranked.rank_in_dept

-- Converted query for original CSV row 2
SELECT
    e.first_name || ' ' || e.last_name AS employee_name,
    to_char(e.hire_date, 'DD-MON-YYYY') AS formatted_hire_date,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.hire_date)) AS tenure_years,
    COALESCE(e.commission_pct, 0) AS commission_value,
    CASE
        WHEN COALESCE(e.commission_pct, 0) = 0 THEN 'Non-Commissioned'
        ELSE 'Commissioned'
    END AS commission_status,
    COALESCE(m.first_name || ' ' || m.last_name, 'No Manager') AS manager_name
FROM
    employees e
LEFT JOIN
    employees m ON e.manager_id = m.employee_id
ORDER BY
    tenure_years DESC

-- Converted query for original CSV row 3
SELECT
    p.project_name,
    COUNT(CASE WHEN ep.role = 'Lead Developer' THEN ep.employee_id END) AS lead_developers,
    COUNT(CASE WHEN ep.role = 'QA Engineer' THEN ep.employee_id END) AS qa_engineers,
    COUNT(CASE WHEN ep.role = 'Project Manager' THEN ep.employee_id END) AS project_managers,
    COUNT(CASE WHEN ep.role NOT IN ('Lead Developer', 'QA Engineer', 'Project Manager') OR ep.role IS NULL THEN ep.employee_id END) AS other_roles
FROM
    projects p
LEFT JOIN
    employee_projects ep ON p.project_id = ep.project_id
GROUP BY
    p.project_name
ORDER BY
    p.project_name

-- Converted query for original CSV row 4
MERGE INTO employees TGT
USING (
    SELECT
        'Emily' AS first_name,
        'Davis' AS last_name,
        'emily.davis@example.com' AS email,
        '555-0106' AS phone_number,
        '2022-01-15'::DATE AS hire_date,
        'SENIOR_ENGINEER' AS job_id,
        82000 AS salary,
        NULL AS commission_pct,
        (SELECT employee_id FROM employees WHERE email = 'alice.johnson@example.com') AS manager_id,
        (SELECT department_id FROM departments WHERE department_name = 'Engineering') AS department_id
) SRC
ON (TGT.email = SRC.email)
WHEN MATCHED THEN
    UPDATE SET
        salary = SRC.salary,
        job_id = SRC.job_id,
        phone_number = SRC.phone_number
WHEN NOT MATCHED THEN
    INSERT (first_name, last_name, email, phone_number, hire_date, job_id, salary, commission_pct, manager_id, department_id)
    VALUES (SRC.first_name, SRC.last_name, SRC.email, SRC.phone_number, SRC.hire_date, SRC.job_id, SRC.salary, SRC.commission_pct, SRC.manager_id, SRC.department_id)

-- Converted query for original CSV row 5
INSERT INTO employees (
    first_name,
    last_name,
    email,
    phone_number,
    hire_date,
    job_id,
    salary,
    commission_pct,
    manager_id,
    department_id
)
VALUES (
    'Kevin',
    'Spacey',
    'kevin.spacey@example.com',
    '555-0199',
    NOW(),
    'ARCHITECT',
    120000,
    NULL,
    (SELECT employee_id FROM employees WHERE email = 'alice.johnson@example.com'),
    (SELECT department_id FROM departments WHERE department_name = 'Engineering')
)
ON CONFLICT (email)
DO UPDATE SET
    salary = EXCLUDED.salary,
    job_id = EXCLUDED.job_id

-- Converted query for original CSV row 6
SELECT
    e.first_name || ' ' || e.last_name AS employee_name,
    e.salary,
    d.department_name,
    e.hire_date,
    SUM(e.salary) OVER (PARTITION BY e.department_id ORDER BY e.hire_date, e.employee_id ROWS UNBOUNDED PRECEDING) AS running_total_salary,
    e.salary - AVG(e.salary) OVER (PARTITION BY e.department_id) AS diff_from_dept_avg_salary
FROM
    employees e
JOIN
    departments d ON e.department_id = d.department_id
WHERE
    d.department_name = 'Engineering'
ORDER BY
    d.department_name, e.hire_date, e.employee_id

-- Converted query for original CSV row 7
SELECT
    d.department_name,
    STRING_AGG(e.first_name || ' ' || e.last_name, ', ' ORDER BY e.hire_date) AS employee_list
FROM
    departments d
LEFT JOIN
    employees e ON d.department_id = e.department_id
GROUP BY
    d.department_id, d.department_name
ORDER BY
    d.department_name

