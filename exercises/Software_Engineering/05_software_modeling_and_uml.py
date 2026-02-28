"""
Exercises for Lesson 05: Software Modeling and UML
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
All UML diagrams are represented as ASCII/text-based descriptions.
"""


# =====================================================================
# PRACTICE EXERCISES (Section 12)
# =====================================================================

# === Practice Exercise 1: Use Case Diagram ===
# Problem: Hotel booking system — 3 actors, 6+ use cases, include/extend.

def practice_exercise_1():
    """Use case diagram for a hotel booking system."""

    diagram = """
USE CASE DIAGRAM: Hotel Booking System
========================================

ACTORS:
  [Guest]  [Receptionist]  [System Administrator]

USE CASES AND RELATIONSHIPS:

  +--------------------------------------------------+
  |           Hotel Booking System                     |
  |                                                    |
  |  (Search Rooms)                                    |
  |  (Book Room) ----<<include>>---> (Authenticate)    |
  |  (Cancel Booking) --<<include>>-> (Authenticate)   |
  |  (Check In Guest)                                  |
  |  (Check Out Guest)                                 |
  |  (Manage Room Inventory)                           |
  |  (Generate Reports)                                |
  |  (Book Room) <---<<extend>>--- (Apply Discount)    |
  |          [condition: loyalty member]                |
  +--------------------------------------------------+

ACTOR-USE CASE ASSOCIATIONS:
  Guest -----------> Search Rooms
  Guest -----------> Book Room
  Guest -----------> Cancel Booking
  Receptionist ----> Check In Guest
  Receptionist ----> Check Out Guest
  Receptionist ----> Search Rooms
  Receptionist ----> Book Room (on behalf of guest)
  SysAdmin --------> Manage Room Inventory
  SysAdmin --------> Generate Reports

INCLUDE RELATIONSHIPS:
  - Book Room <<include>> Authenticate
    (Every booking requires authentication -- mandatory)
  - Cancel Booking <<include>> Authenticate
    (Must verify identity before allowing cancellation)

EXTEND RELATIONSHIPS:
  - Book Room <<extend>> Apply Discount
    (Extension point: loyalty member detected during booking)
"""
    print(diagram)

    use_case_description = """
USE CASE: Book Room
====================
  Actor:          Guest (primary), Receptionist (secondary)
  Preconditions:  Guest is authenticated. At least one room of the
                  desired type is available for the requested dates.
  Trigger:        Guest selects dates and room type and clicks "Book."

  Main Success Scenario:
  1. Guest selects check-in date, check-out date, and room type.
  2. System searches available rooms matching the criteria.
  3. System displays available rooms with prices.
  4. Guest selects a room and confirms.
  5. System reserves the room and generates a booking reference.
  6. System sends confirmation email to the guest.
  7. System updates room availability in real time.

  Alternative Flows:
  3a. No rooms available for the selected dates:
      3a.1. System displays "No rooms available" message.
      3a.2. System suggests alternative dates (nearest availability).
      3a.3. Guest modifies dates and returns to step 2.

  5a. Payment fails:
      5a.1. System displays "Payment failed" error.
      5a.2. Room reservation is not created.
      5a.3. Guest is prompted to try another payment method.

  Postconditions: Room is reserved, booking reference is created,
                  confirmation email is sent.
"""
    print(use_case_description)


# === Practice Exercise 2: Class Diagram ===
# Problem: University course registration system with 5 classes.

def practice_exercise_2():
    """Class diagram for a university course registration system."""

    diagram = """
CLASS DIAGRAM: University Course Registration System
======================================================

INHERITANCE HIERARCHY:
  Student (abstract)
    |-- UndergraduateStudent
    |-- GraduateStudent

CLASS DEFINITIONS:

  +-------------------------------------------+
  |           <<abstract>> Student             |
  +-------------------------------------------+
  | - studentId: String                        |
  | - name: String                             |
  | - email: String                            |
  | - gpa: float                               |
  | - enrollmentDate: Date                     |
  +-------------------------------------------+
  | + enroll(course: Course): Enrollment       |
  | + drop(enrollment: Enrollment): void       |
  | + getTranscript(): List<Enrollment>        |
  +-------------------------------------------+

  +-------------------------------------------+
  |        UndergraduateStudent               |
  +-------------------------------------------+
  | - major: String                            |
  | - year: int (1-4)                          |
  | - advisorId: String                        |
  | - maxCredits: int = 18                     |
  | - dormRoom: String                         |
  +-------------------------------------------+
  | + declareMajor(dept: Department): void     |
  | + checkGraduationEligibility(): boolean    |
  | + getRemainingCredits(): int               |
  +-------------------------------------------+

  +-------------------------------------------+
  |          GraduateStudent                   |
  +-------------------------------------------+
  | - thesisTitle: String                      |
  | - advisor: Instructor                      |
  | - researchArea: String                     |
  | - fundingSource: String                    |
  | - maxCredits: int = 12                     |
  +-------------------------------------------+
  | + submitThesisProposal(): void             |
  | + getResearchGroup(): List<GradStudent>    |
  | + applyForFunding(source: String): boolean |
  +-------------------------------------------+

  +-------------------------------------------+
  |              Course                        |
  +-------------------------------------------+
  | - courseId: String                         |
  | - title: String                            |
  | - credits: int                             |
  | - maxCapacity: int                         |
  | - schedule: String                         |
  +-------------------------------------------+
  | + getAvailableSeats(): int                 |
  | + isFull(): boolean                        |
  | + getEnrolledStudents(): List<Student>     |
  +-------------------------------------------+

  +-------------------------------------------+
  |            Enrollment                      |
  +-------------------------------------------+
  | - enrollmentId: String                     |
  | - enrollDate: Date                         |
  | - grade: String (nullable)                 |
  | - status: EnrollmentStatus                 |
  | - semester: String                         |
  +-------------------------------------------+
  | + assignGrade(grade: String): void         |
  | + withdraw(): void                         |
  | + isActive(): boolean                      |
  +-------------------------------------------+

  +-------------------------------------------+
  |            Instructor                      |
  +-------------------------------------------+
  | - instructorId: String                     |
  | - name: String                             |
  | - email: String                            |
  | - rank: String                             |
  | - officeHours: String                      |
  +-------------------------------------------+
  | + assignToCourse(course: Course): void     |
  | + submitGrades(enrollment: Enrollment): void|
  | + getCourseLoad(): int                     |
  +-------------------------------------------+

  +-------------------------------------------+
  |           Department                       |
  +-------------------------------------------+
  | - deptId: String                           |
  | - name: String                             |
  | - building: String                         |
  | - budget: float                            |
  | - headOfDept: Instructor                   |
  +-------------------------------------------+
  | + addCourse(course: Course): void          |
  | + removeCourse(courseId: String): void      |
  | + listInstructors(): List<Instructor>      |
  +-------------------------------------------+

RELATIONSHIPS:
  Student ---enrolls in---> Course   (through Enrollment, many-to-many)
    Student 1 ----> * Enrollment
    Course  1 ----> * Enrollment

  Instructor 1 ---teaches---> * Course
  Department  1 ---contains---> * Course          (COMPOSITION: courses cannot
                                                   exist without a department)
  Department  1 ---employs---> * Instructor        (AGGREGATION: instructors can
                                                   exist independently)
  GraduateStudent * ---advised by---> 1 Instructor

MULTIPLICITY SUMMARY:
  Student     1    ---- *    Enrollment
  Course      1    ---- *    Enrollment
  Instructor  1    ---- *    Course
  Department  1    ---- *    Course         (composition)
  Department  1    ---- *    Instructor     (aggregation)
  Instructor  1    ---- *    GraduateStudent (advisor)

COMPOSITION vs AGGREGATION JUSTIFICATION:
  - Department ◆--- Course: COMPOSITION. If the CS department is dissolved,
    its courses cease to exist as offered courses. The department OWNS its courses.
  - Department ◇--- Instructor: AGGREGATION. If a department is dissolved,
    instructors still exist -- they transfer to another department.
"""
    print(diagram)


# === Practice Exercise 3: Sequence Diagram ===
# Problem: "Student enrolls in a course" with alt and loop/opt fragments.

def practice_exercise_3():
    """Sequence diagram for student enrollment."""

    diagram = """
SEQUENCE DIAGRAM: Student Enrolls in a Course
===============================================

  Student       WebUI      EnrollmentService    CourseDB      NotificationService
    |             |              |                  |                |
    |--selectCourse(CS101)-->|  |                  |                |
    |             |--enroll(studentId,CS101)-->|    |                |
    |             |              |--checkPrereqs(studentId,CS101)-->|
    |             |              |<--prereqsOK------|                |
    |             |              |                  |                |
    |             |              |--getAvailableSeats(CS101)------->|
    |             |              |<--seats=15--------|                |
    |             |              |                  |                |
    |         +---+-------- [alt] ---------+       |                |
    |         | [seats > 0]                |       |                |
    |         |   |--createEnrollment()----------->|                |
    |         |   |<--enrollmentId=E-4567--|       |                |
    |         |   |              |                  |                |
    |         |   |   +-- [opt: student.email != null] --+          |
    |         |   |   |  |--sendConfirmation(email)----->|          |
    |         |   |   |  |<--sent OK--------------------|          |
    |         |   |   +----------------------------------+          |
    |         |   |              |                  |                |
    |         |   |<--success(E-4567)--|            |                |
    |         |   |              |                  |                |
    |         | [seats == 0]           |            |                |
    |         |   |--addToWaitlist(studentId,CS101)->|               |
    |         |   |<--waitlistPos=3----|            |                |
    |         |   |<--waitlisted(pos=3)|            |                |
    |         +---+--------------------+            |                |
    |             |                     |            |                |
    |<--result---|                     |            |                |
    |             |                     |            |                |

FRAGMENTS:
  alt: Course is full (seats == 0) vs. seats available (seats > 0)
  opt: Send confirmation email only if student has an email on file

OBJECTS INVOLVED: Student (actor), WebUI, EnrollmentService, CourseDB, NotificationService
"""
    print(diagram)


# === Practice Exercise 4: Activity and State Machine ===
# Problem: ATM withdraw cash activity diagram + ATM state machine.

def practice_exercise_4():
    """Activity diagram and state machine for an ATM system."""

    activity = """
ACTIVITY DIAGRAM: ATM Withdraw Cash
======================================

  (●) Start
   |
   v
  [Insert Card]
   |
   v
  [Read Card Data]
   |
   v
  [Prompt for PIN]
   |
   v
  [Enter PIN]
   |
   v
  <Validate PIN> ----[invalid]--> [Increment Attempt Counter]
   |                                        |
   [valid]                            <attempts < 3?> --[yes]--> [Prompt for PIN]
   |                                        |
   v                                    [no / max reached]
  [Prompt for Amount]                       |
   |                                        v
   v                                  [Retain Card]
  [Enter Amount]                            |
   |                                        v
   v                                  [Display: Card Retained]
  <Check Balance> ----[insufficient]--> [Display: Insufficient Funds]
   |                                        |
   [sufficient]                             v
   |                                  [Eject Card]
   v                                        |
  [Dispense Cash]                           v
   |                                      (●) End
   v
  <Print Receipt?> --[yes]--> [Print Receipt]
   |                              |
   [no]                           v
   |<-----------------------------+
   v
  [Eject Card]
   |
   v
  (●) End
"""
    print(activity)

    state_machine = """
STATE MACHINE DIAGRAM: ATM States
====================================

  (●) -----> [Idle]
                |
                | insertCard
                v
           [CardInserted]
                |
                | readCard / display "Enter PIN"
                v
           [PINEntry]
                |
         +------+------+
         |             |
    correctPIN    wrongPIN [attempts < 3]
         |             |
         v             v
   [Authenticated]  [PINEntry]
         |             |
    selectTransaction  wrongPIN [attempts == 3]
         |             |
         v             v
  [TransactionInProgress]  [CardRetained] --> (●) End
         |
    +----+----+
    |         |
  approved  denied
    |         |
    v         v
 [Dispensing]  [TransactionInProgress]
    |             (display error, retry)
    | cashDispensed
    v
 [EjectingCard]
    |
    | cardRemoved
    v
  [Idle]

SPECIAL TRANSITIONS:
  ANY STATE --[hardwareFault]--> [OutOfService]
  [OutOfService] --[technicianReset]--> [Idle]

ENTRY/EXIT ACTIONS:
  [PINEntry]        entry: display "Enter PIN", start 30s timeout
                    exit: clear PIN from memory
  [Dispensing]      entry: lock cash tray, count bills
                    exit: unlock cash tray
  [OutOfService]    entry: display "Out of Service", log error, alert bank
"""
    print(state_machine)


# === Practice Exercise 5: Architecture Diagrams ===
# Problem: Component diagram + deployment diagram for microservices e-commerce.

def practice_exercise_5():
    """Component and deployment diagrams for microservices e-commerce."""

    component = """
COMPONENT DIAGRAM: E-Commerce Microservices Backend
=====================================================

  +---------------------+          +---------------------+
  |    <<component>>    |          |    <<component>>    |
  |    API Gateway      |          |  Order Service      |
  |---------------------|          |---------------------|
  | Provided:           |          | Provided:           |
  |  o REST API (public)|--------->|  o /orders (REST)   |
  |  o Rate Limiting    |          | Required:           |
  |  o Auth Middleware   |          |  * Inventory check  |
  | Required:           |          |  * Payment process  |
  |  * Order API        |          |  * Notification send|
  |  * Inventory API    |          +---------------------+
  |  * Payment API      |                 |           |
  +---------------------+                 |           |
                                          v           v
  +---------------------+    +---------------------+  +---------------------+
  |    <<component>>    |    |    <<component>>    |  |    <<component>>    |
  | Inventory Service   |    |  Payment Service    |  | Notification Service|
  |---------------------|    |---------------------|  |---------------------|
  | Provided:           |    | Provided:           |  | Provided:           |
  |  o /inventory (REST)|    |  o /payments (REST) |  |  o /notify (REST)   |
  |  o Stock check      |    | Required:           |  |  o Email sending    |
  | Required:           |    |  * Stripe API (ext) |  |  o SMS sending      |
  |  * Product DB       |    |  * Payment DB       |  | Required:           |
  +---------------------+    +---------------------+  |  * SendGrid (ext)   |
                                                       +---------------------+

INTERFACES (Provided/Required):
  API Gateway --provides--> Public REST API (HTTP/JSON)
  API Gateway --requires--> Order, Inventory, Payment service APIs
  Order Service --provides--> /orders endpoint
  Order Service --requires--> Inventory check, Payment processing, Notifications
  Payment Service --requires--> External Stripe API
  Notification Service --requires--> External SendGrid/Twilio API
"""
    print(component)

    deployment = """
DEPLOYMENT DIAGRAM: Kubernetes Deployment
==========================================

  +================================================================+
  |                    Kubernetes Cluster                            |
  |                                                                  |
  |  +---------------------------+                                   |
  |  | <<node>> Load Balancer    |                                   |
  |  | (AWS ALB / nginx-ingress) |                                   |
  |  | Port: 443 (HTTPS)        |                                   |
  |  +------------+--------------+                                   |
  |               |                                                  |
  |  +------------v--------------+   +---------------------------+   |
  |  | <<pod>> api-gateway       |   | <<pod>> api-gateway       |   |
  |  | Replicas: 3               |   | (replica 2 of 3)          |   |
  |  | Container: api-gw:v2.1    |   +---------------------------+   |
  |  | Port: 8080               |                                   |
  |  +---------------------------+                                   |
  |               |                                                  |
  |  +------------v--------------+   +---------------------------+   |
  |  | <<pod>> order-service     |   | <<pod>> inventory-service |   |
  |  | Replicas: 2               |   | Replicas: 2               |   |
  |  | Container: order:v1.4     |   | Container: inventory:v1.2 |   |
  |  +---------------------------+   +---------------------------+   |
  |               |                                                  |
  |  +------------v--------------+   +---------------------------+   |
  |  | <<pod>> payment-service   |   | <<pod>> notification-svc  |   |
  |  | Replicas: 2               |   | Replicas: 1               |   |
  |  | Container: payment:v1.1   |   | Container: notify:v1.0    |   |
  |  +---------------------------+   +---------------------------+   |
  |                                                                  |
  +================================================================+

  +========================+     +========================+
  | <<node>> PostgreSQL    |     | <<node>> Redis Cache   |
  | Primary (RDS)          |     | (ElastiCache)          |
  | Host: db-primary       |     | Host: cache.local      |
  | Port: 5432             |     | Port: 6379             |
  +----------+-------------+     +========================+
             |
  +----------v-------------+
  | <<node>> PostgreSQL    |
  | Replica (RDS Read)     |
  | Host: db-replica       |
  | Port: 5432 (read-only) |
  +========================+

COMMUNICATION:
  Load Balancer --> API Gateway Pods (round-robin, port 8080)
  API Gateway --> Service Pods (Kubernetes Service discovery, ClusterIP)
  All services --> PostgreSQL Primary (read/write)
  Read-heavy services --> PostgreSQL Replica (read-only queries)
  Order + Inventory services --> Redis (session cache, stock counts)
"""
    print(deployment)


# =====================================================================
# EXERCISES (End of Lesson)
# =====================================================================

# === Exercise 1: Read and Critique a Use Case Diagram ===
# Problem: Examine ATM use case diagram, draw it, identify mistakes, write prose.

def exercise_1():
    """Critique an ATM use case diagram."""

    diagram = """
EXERCISE 1: ATM Use Case Diagram — Critique
=============================================

(a) ASCII REPRESENTATION:

    [Customer]                              [Bank Network]
        |                                        |
        +---> (Insert Card)                      |
        +---> (Withdraw Cash) --<<include>>--> (Enter PIN)
        +---> (Check Balance) --<<include>>--> (Enter PIN)
        +---> (Deposit Cash) ---<<include>>--> (Enter PIN)
        +---> (Change PIN) ----<<include>>--> (Enter PIN)
        |                                        |
        +---> (Enter PIN)      <--- MISTAKE      |
                                                 |
        [Bank Network] ----> (Withdraw Cash)     |
        [Bank Network] ----> (Deposit Cash)      |

(b) THREE MISTAKES / IMPROVEMENTS:

1. MISTAKE: Customer should NOT be directly associated with "Enter PIN."
   "Enter PIN" is an included use case -- it is invoked ONLY as part of
   other use cases (Withdraw, Balance, Deposit, Change PIN). Directly
   associating the actor with it implies it is a standalone action, which
   makes no sense ("I want to enter my PIN" is not a user goal).
   FIX: Remove the direct association between Customer and Enter PIN.

2. MISTAKE: "Insert Card" is a system step, not a use case.
   A use case should represent a user GOAL. "Insert Card" is a mechanical
   step within every other use case, not a goal in itself.
   FIX: Remove "Insert Card" as a use case. Card insertion is part of the
   precondition or first step of each use case.

3. IMPROVEMENT: Bank Network's role is unclear.
   The diagram associates Bank Network with Withdraw and Deposit but not
   with Check Balance or Change PIN. Bank Network should be associated with
   ALL use cases that require account verification (all four).
   FIX: Associate Bank Network with Check Balance and Change PIN as well,
   or clarify that Bank Network is only needed for fund transfers.

(c) PROSE USE CASE: Withdraw Cash

USE CASE: Withdraw Cash
  Actors: Customer (primary), Bank Network (secondary)
  Preconditions:
    - ATM is operational (not Out of Service)
    - Customer has a valid bank card

  Main Success Scenario:
  1. Customer inserts bank card.
  2. ATM reads card data and validates card format.
  3. ATM prompts for PIN.    [<<include>> Enter PIN]
  4. Customer enters 4-digit PIN.
  5. ATM sends card + PIN to Bank Network for verification.
  6. Bank Network confirms authentication.
  7. ATM displays transaction menu; Customer selects "Withdraw."
  8. ATM prompts for amount.
  9. Customer enters amount (e.g., $100).
  10. ATM sends withdrawal request to Bank Network.
  11. Bank Network verifies sufficient balance and approves.
  12. ATM dispenses cash.
  13. ATM prints receipt.
  14. ATM ejects card.

  Alternative Flow 1: Invalid PIN (step 6)
  6a. Bank Network rejects PIN.
  6b. ATM displays "Invalid PIN. Please try again."
  6c. Return to step 3 (max 3 attempts).
  6d. After 3 failures, ATM retains card and displays error message.

  Alternative Flow 2: Insufficient Funds (step 11)
  11a. Bank Network rejects: insufficient balance.
  11b. ATM displays "Insufficient funds. Your balance is $X."
  11c. Return to step 8 (customer may enter a smaller amount).
"""
    print(diagram)


# === Exercise 2: Hospital Class Diagram ===
# Problem: Design class diagram for hospital patient management.

def exercise_2():
    """Class diagram for a hospital patient management system."""

    diagram = """
EXERCISE 2: Hospital Patient Management System — Class Diagram
================================================================

  +-------------------------------------------+
  |         <<abstract>> Person               |
  +-------------------------------------------+
  | - personId: String                        |
  | - name: String                            |
  | - email: String                           |
  | - phone: String                           |
  | - dateOfBirth: Date                       |
  +-------------------------------------------+
  | + getAge(): int                           |
  | + getContactInfo(): String                |
  +-------------------------------------------+
         ^                    ^
         |                    |
  +------+------+     +------+------+
  |             |     |             |
  |   Patient   |     |   <<abstract>> Staff  |
  +-------------+     +-------------+
  (see below)         ^            ^
                      |            |
               +------+    +------+
               |            |
            Doctor        Nurse

  +-------------------------------------------+
  |              Patient                       |
  +-------------------------------------------+
  | - patientId: String                        |
  | - insuranceNumber: String                  |
  | - bloodType: String                        |
  | - emergencyContact: String                 |
  | - admissionDate: Date                      |
  +-------------------------------------------+
  | + getActiveAppointments(): List<Appointment>|
  | + getMedicalHistory(): List<MedicalRecord>  |
  +-------------------------------------------+

  +-------------------------------------------+
  |               Doctor                       |
  +-------------------------------------------+
  | - licenseNumber: String                    |
  | - specialization: String                   |
  | - consultationFee: float                   |
  | - officeNumber: String                     |
  | - availableSlots: List<TimeSlot>           |
  +-------------------------------------------+
  | + diagnose(patient: Patient): MedicalRecord|
  | + prescribe(rx: Prescription): void        |
  +-------------------------------------------+

  +-------------------------------------------+
  |               Nurse                        |
  +-------------------------------------------+
  | - certificationLevel: String               |
  | - shift: String                            |
  | - assignedWard: String                     |
  | - skills: List<String>                     |
  | - supervisorId: String                     |
  +-------------------------------------------+
  | + recordVitals(patient: Patient): void     |
  | + administerMedication(rx: Prescription): void |
  +-------------------------------------------+

  +-------------------------------------------+
  |           Appointment                      |
  +-------------------------------------------+
  | - appointmentId: String                    |
  | - dateTime: DateTime                       |
  | - duration: int (minutes)                  |
  | - status: AppointmentStatus                |
  | - notes: String                            |
  +-------------------------------------------+
  | + reschedule(newDateTime: DateTime): void  |
  | + cancel(): void                           |
  +-------------------------------------------+

  +-------------------------------------------+
  |          MedicalRecord                     |
  +-------------------------------------------+
  | - recordId: String                         |
  | - date: Date                               |
  | - diagnosis: String                        |
  | - treatment: String                        |
  | - followUpDate: Date                       |
  +-------------------------------------------+
  | + addNote(note: String): void              |
  | + isActive(): boolean                      |
  +-------------------------------------------+

  +-------------------------------------------+
  |           Department                       |
  +-------------------------------------------+
  | - deptId: String                           |
  | - name: String                             |
  | - floor: int                               |
  | - headDoctor: Doctor                       |
  | - budget: float                            |
  +-------------------------------------------+
  | + addStaff(staff: Staff): void             |
  | + getPatientCount(): int                   |
  +-------------------------------------------+

  +-------------------------------------------+
  |          Prescription                      |
  +-------------------------------------------+
  | - rxId: String                             |
  | - medication: String                       |
  | - dosage: String                           |
  | - frequency: String                        |
  | - startDate: Date                          |
  +-------------------------------------------+
  | + isExpired(): boolean                     |
  | + refill(): Prescription                   |
  +-------------------------------------------+

RELATIONSHIPS:
  Patient    1 -----> * Appointment (association)
  Doctor     1 -----> * Appointment (association)
  Patient    1 ◆----> * MedicalRecord (COMPOSITION: records belong to patient)
  MedicalRecord 1 --> * Prescription (COMPOSITION: prescriptions belong to record)
  Department 1 ◇----> * Doctor (AGGREGATION: doctors exist independently)
  Department 1 ◇----> * Nurse  (AGGREGATION: nurses exist independently)

COMPOSITION vs AGGREGATION:
  - Patient ◆ MedicalRecord: COMPOSITION — a medical record is meaningless
    without its patient. If the patient record is deleted, their records go too.
  - MedicalRecord ◆ Prescription: COMPOSITION — prescriptions are created as
    part of a medical record and do not exist independently.
  - Department ◇ Doctor: AGGREGATION — a doctor exists as a professional
    independent of any single department. They can transfer departments.
"""
    print(diagram)


# === Exercise 3: Sequence Diagram ===
# Problem: "Patient books an appointment" with alt and opt fragments.

def exercise_3():
    """Sequence diagram for patient booking an appointment."""

    diagram = """
EXERCISE 3: Patient Books an Appointment — Sequence Diagram
=============================================================

  Patient    Website    AppointmentService   DoctorSchedule   NotificationService
    |          |              |                    |                  |
    |--selectDoctor("Dr.Lee")-->|                  |                  |
    |          |--checkAvailability("Dr.Lee",date)-->|                |
    |          |              |--getSlots("Dr.Lee",date)------------>|
    |          |              |<--availableSlots=[9am,11am,2pm]------|
    |          |              |                    |                  |
    |<--displaySlots([9am,11am,2pm])--|            |                  |
    |          |              |                    |                  |
    |--selectSlot(9am)------->|                    |                  |
    |          |--bookAppointment(patientId,"Dr.Lee",date,9am)-->|   |
    |          |              |                    |                  |
    |       +--+--------- [alt] ----------+       |                  |
    |       | [doctor is available at 9am] |       |                  |
    |       |  |--reserveSlot("Dr.Lee",date,9am)-->|                 |
    |       |  |<--confirmed(apptId=A-789)---------|                 |
    |       |  |              |                    |                  |
    |       |  | +-- [opt: patient.email is not null] --+            |
    |       |  | |   |--sendConfirmation(email,A-789)-->|            |
    |       |  | |   |<--emailSent OK-------------------|            |
    |       |  | +------------------------------------------+        |
    |       |  |              |                    |                  |
    |       |  |<--success(A-789)--|               |                  |
    |       |  |              |                    |                  |
    |       | [doctor is fully booked]             |                  |
    |       |  |<--error("No slots available")--|  |                  |
    |       |  |--suggestAlternatives("Dr.Lee")-->|                  |
    |       |  |<--nextAvailable=["tomorrow 10am"]|                  |
    |       |  |<--fullyBooked(suggestions)--|     |                  |
    |       +--+------------------------------+   |                  |
    |          |                               |   |                  |
    |<--result(success or suggestions)--|       |   |                  |
    |          |                               |   |                  |

LIFELINES (5): Patient, Website, AppointmentService, DoctorSchedule, NotificationService
FRAGMENTS:
  alt: Doctor available (create appointment) vs fully booked (suggest alternatives)
  opt: Send confirmation email only if patient has email address on file
RETURN MESSAGES: Every synchronous call has a return (confirmed, emailSent, etc.)
"""
    print(diagram)


# === Exercise 4: Activity Diagram with Swimlanes ===
# Problem: Hospital discharge process with fork/join and decision.

def exercise_4():
    """Activity diagram for hospital discharge process."""

    diagram = """
EXERCISE 4: Hospital Discharge Process — Activity Diagram with Swimlanes
==========================================================================

  Doctor          |    Nurse         |    Pharmacy      |    Billing
  ================|==================|==================|=================
                  |                  |                  |
  (●) Start       |                  |                  |
   |              |                  |                  |
   v              |                  |                  |
  [Approve        |                  |                  |
   Discharge]     |                  |                  |
   |              |                  |                  |
   |========= FORK (parallel) ==========================================
   |              |                  |                  |
   v              v                  v                  v
  [Review         [Prepare           [Prepare           [Generate
   Follow-up      Discharge          Take-Home          Final
   Needs]         Summary]           Medication]        Invoice]
   |              |                  |                  |
   |              v                  |                  |
   |              [Collect Final     |                  |
   |               Vitals]          |                  |
   |              |                  |                  |
   |========= JOIN (synchronize) =======================================
   |              |                  |                  |
   v              |                  |                  |
  <Follow-up      |                  |                  |
   Needed?>       |                  |                  |
   |         |    |                  |                  |
  [yes]    [no]   |                  |                  |
   |         |    |                  |                  |
   v         |    |                  |                  |
  [Schedule  |    |                  |                  |
   Follow-up |    |                  |                  |
   Appt]     |    |                  |                  |
   |         |    |                  |                  |
   v<--------+    |                  |                  |
   |              |                  |                  |
   |--- hand off to -->|             |                  |
   |              v                  |                  |
   |              [Patient Signs     |                  |
   |               Discharge Form]   |                  |
   |              |                  |                  |
   |              v                  |                  |
   |              [Patient Leaves]   |                  |
   |              |                  |                  |
   |              v                  |                  |
   |              (●) End            |                  |

ELEMENTS:
  FORK: After doctor approves discharge, four activities happen in parallel:
    1. Doctor reviews follow-up needs
    2. Nurse prepares discharge summary + collects vitals
    3. Pharmacy prepares take-home medication
    4. Billing generates final invoice
  JOIN: All four parallel activities must complete before proceeding
  DECISION: Does the patient need a follow-up appointment?
    [yes] → Doctor schedules follow-up
    [no]  → Proceed directly to discharge signing
"""
    print(diagram)


# === Exercise 5: State Machine for Document Workflow ===
# Problem: Complete state machine with entry/exit actions, guards, composite states.

def exercise_5():
    """State machine for a document workflow system."""

    diagram = """
EXERCISE 5: Document Workflow — State Machine Diagram
=======================================================

Choice: (b) Document workflow (Draft -> Review -> Approved -> Published)

  (●) ----create()----> [Draft]
                          |
                          | entry/ assign author, set createdDate
                          | exit/ validate required fields present
                          |
                     submitForReview()
                     [guard: all required fields filled]
                          |
                          v
              +========================+
              |   <<composite>>        |
              |     [InReview]         |
              |                        |
              |  (●)-->[PendingReview] |
              |          |             |
              |    startReview()       |
              |    [guard: reviewer    |
              |     != author]        |
              |          v             |
              |    [UnderReview]       |
              |     |           |      |
              |  approve()  requestChanges()
              |     |           |      |
              |     v           v      |
              |  [Reviewed]  [ChangesRequested] --resubmit()--> [PendingReview]
              |                        |
              +========================+
                          |
                     approve() [from Reviewed]
                          |
                          v
                      [Approved]
                          |
                          | entry/ notify stakeholders, lock content
                          | exit/ generate version number
                          |
                     publish()
                     [guard: legal sign-off == true]
                          |
                          v
                      [Published]
                          |
                          | entry/ set publishedDate, make public
                          |
                     +----+----+
                     |         |
                 archive()  unpublish()
                     |         |
                     v         v
                [Archived]  [Draft]
                     |       (new version)
                     v
                    (●) End

ANY STATE ---[except Published]--- requestDelete() ---> [Deleted] --> (●) End

TRANSITIONS SUMMARY:
  Draft --submitForReview()--> InReview.PendingReview
    Guard: all required fields filled (title, body, category)

  PendingReview --startReview()--> UnderReview
    Guard: reviewer is not the author (separation of duties)

  UnderReview --approve()--> Reviewed
  UnderReview --requestChanges()--> ChangesRequested
  ChangesRequested --resubmit()--> PendingReview
  Reviewed --approve()--> Approved (exits composite state)
  Approved --publish()--> Published
    Guard: legal sign-off is true

  Published --archive()--> Archived
  Published --unpublish()--> Draft (creates new draft version)

ENTRY/EXIT ACTIONS:
  Draft:      entry = assign author; exit = validate fields
  InReview:   entry = notify reviewers; exit = record review outcome
  Approved:   entry = notify stakeholders, lock edits; exit = generate version
  Published:  entry = set publishedDate, make content publicly visible

NON-DETERMINISM WARNING:
  In the UnderReview state, if the guard on approve() and requestChanges()
  were both omitted, a reviewer could theoretically fire both transitions
  simultaneously (e.g., via two concurrent API calls), leading to the
  document being in both Reviewed AND ChangesRequested states. The guards
  (only one reviewer action per review cycle, enforced by state) prevent
  this non-determinism.
"""
    print(diagram)


if __name__ == "__main__":
    print("=" * 65)
    print("=== PRACTICE EXERCISES (Section 12) ===")
    print("=" * 65)

    exercises = [
        ("Practice Exercise 1: Use Case Diagram", practice_exercise_1),
        ("Practice Exercise 2: Class Diagram", practice_exercise_2),
        ("Practice Exercise 3: Sequence Diagram", practice_exercise_3),
        ("Practice Exercise 4: Activity + State Machine", practice_exercise_4),
        ("Practice Exercise 5: Architecture Diagrams", practice_exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\n\n" + "=" * 65)
    print("=== EXERCISES (End of Lesson) ===")
    print("=" * 65)

    end_exercises = [
        ("Exercise 1: Critique Use Case Diagram", exercise_1),
        ("Exercise 2: Hospital Class Diagram", exercise_2),
        ("Exercise 3: Sequence Diagram", exercise_3),
        ("Exercise 4: Activity Diagram with Swimlanes", exercise_4),
        ("Exercise 5: Document Workflow State Machine", exercise_5),
    ]
    for title, func in end_exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
