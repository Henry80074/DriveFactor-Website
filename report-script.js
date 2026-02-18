// Mobile menu functionality
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const navLinks = document.getElementById('navLinks');

if (mobileMenuBtn && navLinks) {
    mobileMenuBtn.addEventListener('click', function() {
        navLinks.classList.toggle('active');
        mobileMenuBtn.classList.toggle('active');
    });

    // Close mobile menu when clicking on a link
    navLinks.addEventListener('click', function(e) {
        if (e.target.tagName === 'A') {
            navLinks.classList.remove('active');
            mobileMenuBtn.classList.remove('active');
        }
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', function(e) {
        if (!navLinks.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
            navLinks.classList.remove('active');
            mobileMenuBtn.classList.remove('active');
        }
    });
}

// Spider Chart
const ctx = document.getElementById('spiderChart').getContext('2d');
const spiderChart = new Chart(ctx, {
    type: 'radar',
    data: {
        labels: [
            'Speed Management',
            'Urban Driving',
            'Traffic Navigation',
            'Weather Adaptation',
            'Night Driving',
            'Highway Performance',
            'Smooth Acceleration',
            'Braking Control'
        ],
        datasets: [{
            label: 'John Davies',
            data: [92, 85, 78, 52, 41, 68, 81, 65],
            fill: true,
            backgroundColor: 'rgba(37, 99, 235, 0.2)',
            borderColor: 'rgb(37, 99, 235)',
            pointBackgroundColor: 'rgb(37, 99, 235)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(37, 99, 235)',
            borderWidth: 2
        }, {
            label: 'Fleet Average',
            data: [75, 72, 70, 68, 65, 70, 73, 72],
            fill: true,
            backgroundColor: 'rgba(100, 116, 139, 0.1)',
            borderColor: 'rgb(100, 116, 139)',
            pointBackgroundColor: 'rgb(100, 116, 139)',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: 'rgb(100, 116, 139)',
            borderWidth: 1,
            borderDash: [5, 5]
        }]
    },
    options: {
        elements: {
            line: {
                borderWidth: 3
            }
        },
        scales: {
            r: {
                angleLines: {
                    display: true
                },
                suggestedMin: 0,
                suggestedMax: 100,
                ticks: {
                    stepSize: 20
                }
            }
        },
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 20,
                    font: {
                        size: 12
                    }
                }
            }
        }
    }
});

// Speed Trend Mini Chart
const speedCtx = document.getElementById('speedTrend').getContext('2d');
new Chart(speedCtx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            data: [84, 87, 90, 92],
            borderColor: 'rgb(16, 185, 129)',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                display: false,
                min: 70,
                max: 100
            },
            x: {
                display: false
            }
        }
    }
});

// Night Performance Trend Mini Chart
const nightCtx = document.getElementById('nightTrend').getContext('2d');
new Chart(nightCtx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            data: [53, 48, 45, 41],
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                display: false,
                min: 30,
                max: 60
            },
            x: {
                display: false
            }
        }
    }
});

// Risk Score Trend Mini Chart
const riskCtx = document.getElementById('riskTrend').getContext('2d');
new Chart(riskCtx, {
    type: 'line',
    data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [{
            data: [52, 53, 54, 54],
            borderColor: 'rgb(245, 158, 11)',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            y: {
                display: false,
                min: 40,
                max: 70
            },
            x: {
                display: false
            }
        }
    }
});

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Print report functionality
function printReport() {
    window.print();
}

// Add print button if needed
const addPrintButton = () => {
    const reportHeader = document.querySelector('.report-header .container');
    if (reportHeader) {
        const printBtn = document.createElement('button');
        printBtn.className = 'btn btn-secondary';
        printBtn.textContent = '🖨️ Print Report';
        printBtn.style.position = 'absolute';
        printBtn.style.top = '20px';
        printBtn.style.right = '20px';
        printBtn.onclick = printReport;
        reportHeader.style.position = 'relative';
        reportHeader.appendChild(printBtn);
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    addPrintButton();
    
    // Add fade-in animation to sections
    const sections = document.querySelectorAll('section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });
    
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
});
